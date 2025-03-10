import copy
import math
import time

import numpy as np

from .common import update_last_task
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

        print("Uniform Sampling Coresets")

    def generate_coreset(self, task_X_, task_y_):
        n_ = task_X_.shape[0]
        coreset_size = int(self.replay_buffer_perc * n_)
        indices = np.random.choice(n_, coreset_size, replace=False)
        task_X = task_X_[indices]
        task_y = task_y_[indices]
        self.coresets[self.current_task] = (task_X, task_y)
        print("Generated Uniform Coreset")

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

    def train_epoch(self, task_num, x, y):
        self.train_epoch_(task_num, x, y)
        if len(self.coresets.keys()) > 0:
            for k, v in self.coresets.items():
                self.train_epoch_(k, v[0], v[1])
