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
        self.model_with_gmm_prior_dict = model.state_dict()

        print("Simplified Coresets")

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

    def generate_coreset(self, task_X_, task_y_):
        last_model = self.shared_model_cache["last_task"]
        state_dict = (
            self.model_with_gmm_prior_dict
            if last_model is None
            else copy.deepcopy(
                self.shared_model_cache["models"][last_model].state_dict()
            )
        )
        n_ = task_X_.shape[0]
        indices = np.random.choice(n_, n_, replace=False)
        task_X = task_X_[indices]
        task_y = task_y_[indices]
        coreset_size = int(self.replay_buffer_perc * n_)
        best_loss = np.inf

        for i in range((n_ // coreset_size) - 1):
            candidate_indices = list(range(i * coreset_size, (i + 1) * coreset_size))
            X_subset = task_X[candidate_indices]
            y_subset = task_y[candidate_indices]
            model = load_network_with_args()
            model.load_state_dict(copy.deepcopy(state_dict))
            stub_model_trained = self.train_stub(model, X_subset, y_subset)
            kl_div = self.get_kl_divergence(stub_model_trained, self.model)
            if kl_div < best_loss:
                best_loss = kl_div
                self.coresets[self.current_task] = (X_subset, y_subset)

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
        self.generate_coreset(xtrain, ytrain)
        self.model.load_state_dict(copy.deepcopy(best_model))
        self.save_model(task_num)

    def train_epoch(self, task_num, x, y):
        self.train_epoch_(task_num, x, y)
        if len(self.coresets.keys()) > 0:
            for k, v in self.coresets.items():
                self.train_epoch_(k, v[0], v[1])
