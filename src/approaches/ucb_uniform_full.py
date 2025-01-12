import copy
import math
import random
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

        self.prev_task_data = self.shared_model_cache["prev_task_data"]
        self.task_freq = self.shared_model_cache["task_frquencies"]
        self.replay_buffer_perc = args.rbuff_size

        print("Uniform Sampling at every epoch")

    def get_prob_from_freq(self):
        n = len(self.task_freq.keys())
        probs = {key: 1 / n for key in self.task_freq.keys()}
        return probs

    def sample_prev_task(self, probs):
        keys = list(probs.keys())
        values = list(probs.values())
        return random.choices(keys, weights=values, k=1)[0]

    def get_buffer(self, prev_task, buffer_size):
        prev_xtrain = self.prev_task_data[prev_task]["xtrain"]
        prev_ytrain = self.prev_task_data[prev_task]["ytrain"]
        n_ = prev_xtrain.shape[0]
        if n_ <= buffer_size:
            return prev_xtrain, prev_ytrain
        indices = np.random.choice(n_, buffer_size, replace=False)
        prev_xtrain = prev_xtrain[indices]
        prev_ytrain = prev_ytrain[indices]
        return prev_xtrain, prev_ytrain

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
                if len(self.prev_task_data.keys()) > 0:
                    probs = self.get_prob_from_freq()
                    sampled_task = self.sample_prev_task(probs)
                    self.task_freq[sampled_task] += 1
                    prev_xtrain, prev_ytrain = self.get_buffer(
                        sampled_task, int(self.replay_buffer_perc * xtrain.shape[0])
                    )
                    self.sampled_task = sampled_task
                else:
                    prev_xtrain, prev_ytrain = None, None
                clock0 = time.time()
                self.train_epoch(task_num, xtrain, ytrain, prev_xtrain, prev_ytrain)
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

        self.model.load_state_dict(copy.deepcopy(best_model))
        self.save_model(task_num)

    def train_epoch(self, task_num, x, y, prevx=None, prevy=None):
        self.train_epoch_(task_num, x, y)
        if prevx is not None and prevy is not None:
            self.train_epoch_(self.sampled_task, prevx, prevy)
