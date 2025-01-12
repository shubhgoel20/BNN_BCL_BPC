import copy
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from ..common import *
from ..utils import BayesianSGD


class Appr(object):

    def __init__(
        self, model, args, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=1000
    ):
        print("UCB New3")
        self.model = model
        self.device = args.device
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.init_lr = args.lr
        self.sbatch = args.sbatch
        self.nepochs = args.nepochs

        self.arch = args.arch
        self.samples = args.samples
        self.lambda_ = 1.0

        self.output = args.output
        self.checkpoint = args.checkpoint
        self.experiment = args.experiment
        self.num_tasks = args.num_tasks

        self.shared_model_cache = shared_model_task_cache

        self.modules_names_with_cls = self.find_modules_names(with_classifier=True)
        self.modules_names_without_cls = self.find_modules_names(with_classifier=False)

        self.coresets = self.shared_model_cache["coresets"]
        self.task_freq = self.shared_model_cache["task_frquencies"]
        self.replay_buffer_perc = args.rbuff_size

        self.model_with_gmm_prior_dict = model.state_dict()

    def get_kl_divergence(self, model1, model2):
        kl_div = 0
        task_num = self.current_task
        for name in shared_model_task_cache["modules_names_without_cls"]:
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

    def generate_coreset(self, task_X_, task_y_):
        # coreset_size = shared_model_task_cache["args"].coreset_size
        last_model = shared_model_task_cache["last_task"]
        state_dict = (
            self.model_with_gmm_prior_dict
            if last_model is None
            else copy.deepcopy(
                shared_model_task_cache["models"][last_model].state_dict()
            )
        )
        n_ = task_X_.shape[0]
        indices = np.random.choice(n_, n_, replace=False)
        task_X = task_X_[indices]
        task_y = task_y_[indices]

        # remaining_indices = list(range(len(task_X)))
        # coreset_indices = []
        coreset_size = int(self.replay_buffer_perc * n_)

        # for _ in range(coreset_size):
        # best_idx = None
        best_loss = np.inf

        for i in range((n_ // coreset_size) - 1):
            # candidate_indices = coreset_indices + [idx]
            candidate_indices = list(range(i * coreset_size, (i + 1) * coreset_size))
            X_subset = task_X[candidate_indices]
            y_subset = task_y[candidate_indices]
            model = load_network_with_args()
            model.load_state_dict(copy.deepcopy(state_dict))
            stub_model_trained = self.train_stub(model, X_subset, y_subset)
            kl_div = self.get_kl_divergence(stub_model_trained, self.model)
            if kl_div < best_loss:
                best_loss = kl_div
                # best_idx = idx
                print(f"Possible coreset entry found: {i} : {best_loss}")
                self.coresets[self.current_task] = (X_subset, y_subset)
            # coreset_indices.append(best_idx)
            # remaining_indices.remove(best_idx)
            # print(f"Coreset entry added: {best_idx} : {best_loss}")
        # self.coresets[self.current_task] = (task_X[coreset_indices], task_y[coreset_indices])

    def train_stub(self, model, xtrain, ytrain):
        params_dict = self.get_model_params()
        self.optimizer = BayesianSGD(params=params_dict)
        try:
            for _ in range(self.nepochs):
                self.train_epoch_(self.current_task, xtrain, ytrain, model_=model)
        except KeyboardInterrupt:
            pass
        return model

    def train(self, task_num, xtrain, ytrain, xvalid, yvalid):

        # Update the next learning rate for each parameter based on their uncertainty
        update_last_task(task_num)
        self.current_task = task_num
        params_dict = self.get_model_params()
        self.optimizer = BayesianSGD(params=params_dict)

        best_loss = np.inf

        # best_model=copy.deepcopy(self.model)
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
        self.generate_coreset(xtrain, ytrain)
        self.model.load_state_dict(copy.deepcopy(best_model))
        self.save_model(task_num)

    def get_model_params(self, model_=None):
        params_dict = []
        if not model_:
            params_dict.append({"params": self.model.parameters(), "lr": self.init_lr})
        else:
            params_dict.append({"params": model_.parameters(), "lr": self.init_lr})
        return params_dict

    def update_lr(self, task_num, lr=None):
        params_dict = []
        if task_num == 0:
            params_dict.append({"params": self.model.parameters(), "lr": self.init_lr})
        else:
            for name in self.modules_names_without_cls:
                n = name.split(".")
                if len(n) == 1:
                    m = self.model._modules[n[0]]
                elif len(n) == 3:
                    m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]
                elif len(n) == 4:
                    m = (
                        self.model._modules[n[0]]
                        ._modules[n[1]]
                        ._modules[n[2]]
                        ._modules[n[3]]
                    )
                else:
                    print(name)
                params_dict.append({"params": m.weight_rho, "lr": lr})
                params_dict.append({"params": m.bias_rho, "lr": lr})

        return params_dict

    def find_modules_names(self, with_classifier=False):
        modules_names = []
        for name, p in self.model.named_parameters():
            if with_classifier is False:
                if not name.startswith("classifier"):
                    n = name.split(".")[:-1]
                    modules_names.append(".".join(n))
            else:
                n = name.split(".")[:-1]
                modules_names.append(".".join(n))

        modules_names = set(modules_names)
        if not with_classifier:
            shared_model_task_cache["modules_names_without_cls"] = modules_names
        return modules_names

    def logs(self, task_num, model_=None):
        if not model_:
            model_ = self.model
        lp_, lvp = 0.0, 0.0
        for name in self.modules_names_without_cls:
            n = name.split(".")
            if len(n) == 1:
                m = model_._modules[n[0]]
            elif len(n) == 3:
                m = model_._modules[n[0]]._modules[n[1]]._modules[n[2]]
            elif len(n) == 4:
                m = model_._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]

            lp_ += m.log_prior
            lvp += m.log_variational_posterior

        lp__, last_model_available = get_log_posterior_from_last_task(model_)
        lp = lp__ if last_model_available else lp_
        lp += model_.classifier[task_num].log_prior
        lvp += model_.classifier[task_num].log_variational_posterior

        return lp, lvp

    def train_epoch(self, task_num, x, y):
        self.train_epoch_(task_num, x, y)
        if len(self.coresets.keys()) > 0:
            for k, v in self.coresets.items():
                self.train_epoch_(k, v[0], v[1])

    def train_epoch_(self, task_num, x, y, model_=None):
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

            # Forward
            loss = self.elbo_loss(
                images, targets, task_num, num_batches, sample=True, model_=model_
            ).to(self.device)

            # Backward
            model_.cuda()
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            model_.cuda()

            # Update parameters
            self.optimizer.step()
        return

    def eval(self, task_num, x, y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.as_tensor(r, device=self.device, dtype=torch.int64)

        with torch.no_grad():
            num_batches = len(x) // self.sbatch
            # Loop batches
            for i in range(0, len(r), self.sbatch):
                if i + self.sbatch <= len(r):
                    b = r[i : i + self.sbatch]
                else:
                    b = r[i:]
                images, targets = x[b].to(self.device), y[b].to(self.device)

                # Forward
                outputs = self.model(images, sample=False)
                output = outputs[task_num]
                loss = self.elbo_loss(
                    images, targets, task_num, num_batches, sample=False
                )

                _, pred = output.max(1, keepdim=True)

                total_loss += loss.detach() * len(b)
                total_acc += pred.eq(targets.view_as(pred)).sum().item()
                total_num += len(b)

        return total_loss / total_num, total_acc / total_num

    def elbo_loss(self, input, target, task_num, num_batches, sample, model_=None):
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
            nll = w3 * torch.nn.functional.nll_loss(
                outputs.mean(0), target, reduction="sum"
            ).to(device=self.device)

            return (log_var - log_p) / num_batches + nll

        else:
            predictions = []
            for i in range(self.samples):
                pred = model_(input, sample=False)[task_num]
                predictions.append(pred)
            w3 = 5.0e-6

            outputs = torch.stack(predictions, dim=0).to(self.device)
            nll = w3 * torch.nn.functional.nll_loss(
                outputs.mean(0), target, reduction="sum"
            ).to(device=self.device)

            return nll

    def save_model(self, task_num):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
            },
            os.path.join(self.checkpoint, "model_{}.pth.tar".format(task_num)),
        )
