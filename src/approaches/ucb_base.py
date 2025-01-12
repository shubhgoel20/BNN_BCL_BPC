import os

import numpy as np
import torch

from .common import shared_model_task_cache


class Approach(object):

    def __init__(
        self, model, args, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=1000
    ):
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

    def save_model(self, task_num):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
            },
            os.path.join(self.checkpoint, "model_{}.pth.tar".format(task_num)),
        )

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
