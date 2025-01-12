import copy
import math
import time

import numpy as np
import torch

from .ucb_base import Approach
from .utils import BayesianSGD

class Appr(Approach):

    def __init__(
        self, model, args, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=1000
    ):
        super().__init__(model, args, lr_min, lr_factor, lr_patience, clipgrad)
        print("UCB Adaptive Learning Rate")

    def train(self, t, xtrain, ytrain, xvalid, yvalid):

        # Update the next learning rate for each parameter based on their uncertainty
        params_dict = self.update_lr(t)
        self.optimizer = BayesianSGD(params=params_dict)

        best_loss = np.inf

        # best_model=copy.deepcopy(self.model)
        best_model = copy.deepcopy(self.model.state_dict())
        lr = self.init_lr
        patience = self.lr_patience

        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0 = time.time()
                self.train_epoch(t, xtrain, ytrain)
                clock1 = time.time()
                train_loss, train_acc = self.eval(t, xtrain, ytrain)
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
                valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
                print(
                    " Valid: loss={:.3f}, acc={:5.1f}% |".format(
                        valid_loss, 100 * valid_acc
                    ),
                    end="",
                )

                if math.isnan(valid_loss) or math.isnan(train_loss):
                    print("saved best model and quit because loss became nan")
                    break

                # Adapt lr
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = copy.deepcopy(self.model.state_dict())
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

                        params_dict = self.update_lr(t, adaptive_lr=True, lr=lr)
                        self.optimizer = BayesianSGD(params=params_dict)

                print()
        except KeyboardInterrupt:
            print()

        # Restore best
        self.model.load_state_dict(copy.deepcopy(best_model))
        self.save_model(t)

    def update_lr(self, t, lr=None, adaptive_lr=False):
        params_dict = []
        if t == 0:
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

                if adaptive_lr is True:
                    params_dict.append({"params": m.weight_rho, "lr": lr})
                    params_dict.append({"params": m.bias_rho, "lr": lr})

                else:
                    w_unc = torch.log1p(torch.exp(m.weight_rho.data))
                    b_unc = torch.log1p(torch.exp(m.bias_rho.data))

                    params_dict.append(
                        {"params": m.weight_mu, "lr": torch.mul(w_unc, self.init_lr)}
                    )
                    params_dict.append(
                        {"params": m.bias_mu, "lr": torch.mul(b_unc, self.init_lr)}
                    )
                    params_dict.append({"params": m.weight_rho, "lr": self.init_lr})
                    params_dict.append({"params": m.bias_rho, "lr": self.init_lr})

        return params_dict

    def logs(self, t, model_=None):

        lp, lvp = 0.0, 0.0
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

            lp += m.log_prior
            lvp += m.log_variational_posterior

        lp += self.model.classifier[t].log_prior
        lvp += self.model.classifier[t].log_variational_posterior

        return lp, lvp

    def train_epoch(self, t, x, y):

        self.model.train()

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).to(self.device)

        num_batches = len(x) // self.sbatch
        j = 0
        # Loop batches
        for i in range(0, len(r), self.sbatch):

            if i + self.sbatch <= len(r):
                b = r[i : i + self.sbatch]
            else:
                b = r[i:]
            images, targets = x[b].to(self.device), y[b].to(self.device)

            # Forward
            loss = self.elbo_loss(images, targets, t, num_batches, sample=True).to(
                self.device
            )

            # Backward
            self.model.cuda()
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.model.cuda()

            # Update parameters
            self.optimizer.step()
        return
