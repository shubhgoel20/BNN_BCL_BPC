import copy
import math
import time

import numpy as np
import torch

from .common import get_log_posterior_from_last_task, update_last_task
from .ucb_base import Approach
from .utils import BayesianSGD


class Appr(Approach):

    def __init__(
        self, model, args, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=1000
    ):
        super().__init__(model, args, lr_min, lr_factor, lr_patience, clipgrad)
        print("UCB Bayesian Continual Learning")

    def train(self, task_num, xtrain, ytrain, xvalid, yvalid):

        update_last_task(task_num)
        params_dict = self.get_model_params()
        self.optimizer = BayesianSGD(params=params_dict)

        best_loss = np.inf

        best_model = copy.deepcopy(self.model.state_dict())
        lr = self.init_lr
        patience = self.lr_patience

        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0 = time.time()
                self.train_epoch_(task_num, xtrain, ytrain)
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
                # Valid
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

        # Restore best
        self.model.load_state_dict(copy.deepcopy(best_model))
        self.save_model(task_num)

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
