import copy
import math
import time

import numpy as np
import torch
from torch.distributions import Normal

from .common import load_network_with_args, update_last_task
from .ucb_bcl import Appr as Approach_BCL
from .utils import BayesianSGD


class Appr(Approach_BCL):

    def __init__(
        self, model, args, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=1000
    ):
        super().__init__(model, args, lr_min, lr_factor, lr_patience, clipgrad)

        self.coresets = self.shared_model_cache["coresets"]
        self.task_freq = self.shared_model_cache["task_frquencies"]
        self.replay_buffer_perc = args.rbuff_size
        self.use_pseudocoreset = args.pseudocoreset
        self.model_with_gmm_prior_dict = copy.deepcopy(model.state_dict())

        print("Bayesian Coresets")

    def get_kl_divergence(self, model1, model2):
        kl_div = 0
        task_num = self.current_task
        for name in self.shared_model_cache["modules_names_without_cls"]:
            n = name.split(".")
            if len(n) == 1:
                model2_ = model2._modules[n[0]]
                model1_ = model1._modules[n[0]]
            elif len(n) == 3:
                model2_ = model2._modules[n[0]]._modules[n[1]]._modules[n[2]]
                model1_ = model1._modules[n[0]]._modules[n[1]]._modules[n[2]]
            elif len(n) == 4:
                model2_ = (
                    model2._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
                )
                model1_ = (
                    model1._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
                )
            model1_weight_dist = Normal(model1_.weight.mu, model1_.weight.sigma)
            model2_weight_dist = Normal(model2_.weight.mu, model2_.weight.sigma)
            kl_div += torch.distributions.kl.kl_divergence(
                model1_weight_dist, model2_weight_dist
            ).sum()
            if model1_.use_bias and model2_.use_bias:
                model1_bias_dist = Normal(model1_.bias.mu, model1_.bias.sigma)
                model2_bias_dist = Normal(model2_.bias.mu, model2_.bias.sigma)
                kl_div += torch.distributions.kl.kl_divergence(
                    model1_bias_dist, model2_bias_dist
                ).sum()

        # Classifier KL Divergence
        model1_weight_dist = Normal(
            model1.classifier[task_num].weight.mu,
            model1.classifier[task_num].weight.sigma,
        )
        model2_weight_dist = Normal(
            model2.classifier[task_num].weight.mu,
            model2.classifier[task_num].weight.sigma,
        )
        kl_div += torch.distributions.kl.kl_divergence(
            model1_weight_dist, model2_weight_dist
        ).sum()
        if model1_.use_bias and model2_.use_bias:
            model1_bias_dist = Normal(
                model1.classifier[task_num].bias.mu,
                model1.classifier[task_num].bias.sigma,
            )
            model2_bias_dist = Normal(
                model2.classifier[task_num].bias.mu,
                model2.classifier[task_num].bias.sigma,
            )
            kl_div += torch.distributions.kl.kl_divergence(
                model1_bias_dist, model2_bias_dist
            ).sum()
        return kl_div

    def generate_coreset(self, task_X_, task_y_, task_num):
        last_task = self.shared_model_cache["last_task"]
        last_model = load_network_with_args()
        if last_task is None:
            last_model.load_state_dict(self.model_with_gmm_prior_dict)
        else:
            last_model.load_state_dict(
                copy.deepcopy(self.shared_model_cache["models"][last_task].state_dict())
            )
        last_model.requires_grad_(False)

        stub_model = load_network_with_args().to(self.device)
        stub_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        stub_model.train()
        self.model.requires_grad_(False)

        n_ = task_X_.shape[0]
        coreset_size = int(self.replay_buffer_perc * n_)
        indices = np.random.choice(n_, coreset_size, replace=False)
        task_X = task_X_[indices]
        task_y = task_y_[indices]
        task_weights = torch.nn.Parameter(torch.ones(coreset_size))

        stub_opt = BayesianSGD(params=self.get_model_params(stub_model))

        coreset_params = [task_weights]
        if self.use_pseudocoreset:
            task_X = torch.nn.Parameter(task_X)
            coreset_params.append(task_X)
        coreset_opt = torch.optim.SGD(coreset_params, lr=1e-2)

        print("Generating coreset for task", task_num)

        for _ in range(self.nepochs):
            r = np.arange(task_X.size(0))
            np.random.shuffle(r)
            r = torch.LongTensor(r).to(self.device)

            num_batches = len(task_X) // self.sbatch
            # Loop batches
            for i in range(0, len(r), self.sbatch):
                if i + self.sbatch <= len(r):
                    b = r[i : i + self.sbatch]
                else:
                    b = r[i:]
                images, targets = task_X[b].to(self.device), task_y[b].to(self.device)
                weights = task_weights[b].to(self.device)

                # Forward
                w1 = 1.0e-3
                w3 = 5.0e-2

                loss = w1 * (
                    self.get_kl_divergence(stub_model, last_model)
                    + self.get_kl_divergence(stub_model, self.model)
                )  # coeff hparam?

                for j in range(self.samples):
                    outputs = stub_model(images, sample=True)[task_num]
                    loss += (
                        w3
                        * weights
                        @ torch.nn.functional.nll_loss(
                            outputs, targets, reduction="none"
                        )
                        / self.samples
                    )

                logp_coreset_list = []
                for j in range(self.samples):
                    outputs = last_model(images, sample=True)[task_num]
                    logp_coreset_list.append(
                        weights
                        @ torch.nn.functional.nll_loss(
                            outputs, targets, reduction="none"
                        )
                        / self.samples
                    )

                logp_coreset_list = torch.stack(logp_coreset_list)
                loss += w3 * torch.logsumexp(logp_coreset_list, dim=0)

                # Backward
                stub_model.cuda()
                stub_opt.zero_grad()
                coreset_opt.zero_grad()
                loss.backward(retain_graph=True)
                stub_model.cuda()

                # Update parameters
                stub_opt.step()
                coreset_opt.step()

        if self.use_pseudocoreset:
            task_X = task_X.data
        task_weights = task_weights.data
        self.coresets[task_num] = (task_X, task_y, task_weights)
        print("Done: coreset for task", task_num)

        self.model.requires_grad_(True)

    def train(self, task_num, xtrain, ytrain, xvalid, yvalid):

        update_last_task(task_num)
        self.current_task = task_num
        params_dict = self.get_model_params()
        self.optimizer = BayesianSGD(params=params_dict)

        best_loss = np.inf

        best_model = copy.deepcopy(self.model.state_dict())
        lr = self.init_lr
        patience = self.lr_patience

        # Loop epochs
        try:
            for e in range(self.nepochs):
                clock0 = time.time()
                self.train_epoch(task_num, xtrain, ytrain)
                clock1 = time.time()
                train_loss, train_acc = self.eval(task_num, xtrain, ytrain)
                clock2 = time.time()

                print(
                    "| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |".format(
                        e + 1,
                        1000 * self.sbatch * (clock1 - clock0) / xtrain.size(0),
                        1000 * self.sbatch * (clock2 - clock1) / xtrain.size(0),
                        train_loss,
                        100 * train_acc,
                    ),
                    end="",
                )
                valid_loss, valid_acc = self.eval(task_num, xvalid, yvalid)
                print(
                    " Valid: loss={:.3f}, acc={:5.1f}% |".format(
                        valid_loss, 100 * valid_acc
                    ),
                    end="",
                )

                if math.isnan(valid_loss) or math.isnan(train_loss):
                    print("saved best model and quit because loss became nan")
                    break

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = copy.deepcopy(self.model.state_dict())
                    self.shared_model_cache["models"][task_num] = copy.deepcopy(
                        self.model.state_dict()
                    )
                    patience = self.lr_patience
                    print(" *", end="")
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= self.lr_factor
                        print(" lr={:.1e}".format(lr), end="")
                        if lr < self.lr_min:
                            print()
                            break
                        patience = self.lr_patience

                        params_dict = self.update_lr(task_num, lr=lr)
                        self.optimizer = BayesianSGD(params=params_dict)

                print()
        except KeyboardInterrupt:
            print()
        self.task_freq[task_num] = int(1 / self.replay_buffer_perc)
        self.generate_coreset(xtrain, ytrain, task_num)
        self.model.load_state_dict(copy.deepcopy(best_model))
        self.save_model(task_num)

    def train_epoch(self, task_num, x, y):
        self.train_epoch_(task_num, x, y)
        if len(self.coresets.keys()) > 0:
            for k, v in self.coresets.items():
                self.train_epoch_(k, *v)

    def train_epoch_(self, task_num, x, y, weights=None, model_=None):
        if not model_:
            model_ = self.model
        model_.train()

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).to(self.device)

        num_batches = len(x) // self.sbatch
        # Loop batches
        for i in range(0, len(r), self.sbatch):

            if i + self.sbatch <= len(r):
                b = r[i : i + self.sbatch]
            else:
                b = r[i:]
            images, targets = x[b].to(self.device), y[b].to(self.device)

            if weights is None:
                w = None
            else:
                w = weights[b].to(self.device)

            # Forward
            loss = self.elbo_loss(
                images,
                targets,
                task_num,
                num_batches,
                sample=True,
                weights=w,
                model_=model_,
            ).to(self.device)

            # Backward
            model_.cuda()
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            model_.cuda()

            # Update parameters
            self.optimizer.step()
        return

    def elbo_loss(
        self, input, target, task_num, num_batches, sample, weights=None, model_=None
    ):
        if not model_:
            model_ = self.model
        if sample:
            lps, lvps, predictions = [], [], []
            for i in range(self.samples):
                predictions.append(model_(input, sample=sample)[task_num])
                lp, lv = self.logs(task_num, model_=model_)
                lps.append(lp)
                lvps.append(lv)

            # hack
            w1 = 1.0e-3
            w2 = 1.0e-3
            w3 = 5.0e-2

            outputs = torch.stack(predictions, dim=0).to(self.device)
            log_var = w1 * torch.as_tensor(lvps, device=self.device).mean()
            log_p = w2 * torch.as_tensor(lps, device=self.device).mean()
            if weights is None:
                nll = w3 * torch.nn.functional.nll_loss(
                    outputs.mean(0), target, reduction="sum"
                ).to(device=self.device)
            else:
                nll = (
                    w3
                    * weights
                    @ torch.nn.functional.nll_loss(
                        outputs.mean(0), target, reduction="none"
                    ).to(device=self.device)
                )

            return (log_var - log_p) / num_batches + nll

        else:
            predictions = []
            for i in range(self.samples):
                pred = model_(input, sample=False)[task_num]
                predictions.append(pred)
            w3 = 5.0e-6

            outputs = torch.stack(predictions, dim=0).to(self.device)
            if weights is None:
                nll = w3 * torch.nn.functional.nll_loss(
                    outputs.mean(0), target, reduction="sum"
                ).to(device=self.device)
            else:
                nll = (
                    w3
                    * weights
                    @ torch.nn.functional.nll_loss(
                        outputs.mean(0), target, reduction="none"
                    ).to(device=self.device)
                )

            return nll
