import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch

import utils
from approaches.common import shared_model_task_cache

tstart = time.time()

# Arguments
parser = argparse.ArgumentParser(description="xxx")
parser.add_argument("--seed", default=0, type=int, help="(default=%(default)d)")
parser.add_argument("--device", default="cuda:0", type=str, help="gpu id")
parser.add_argument(
    "--experiment",
    default="mnist5",
    type=str,
    required=True,
    choices=["mnist2", "mnist5", "pmnist", "cifar"],
)
parser.add_argument(
    "--approach",
    default="ucb_bpc",
    type=str,
    required=True,
    choices=[
        "ucb_adaptive_lr",
        "ucb_bcl",
        "ucb_uniform_full",
        "ucb_simplified_coresets",
        "ucb_uniform",
        "ucb_bpc",
    ],
    help="Approach to run experiments with",
)
parser.add_argument("--data_path", default="../data/", type=str, help="gpu id")

# Training parameters
parser.add_argument("--output", default="", type=str, help="")
parser.add_argument("--checkpoint_dir", default="../checkpoints/", type=str, help="")
parser.add_argument("--nepochs", default=200, type=int, help="")
parser.add_argument("--sbatch", default=64, type=int, help="")
parser.add_argument(
    "--lr", default=0.01, type=float, help=""
)  # use 0.3 for non-mnist datasets
parser.add_argument("--nlayers", default=1, type=int, help="")
parser.add_argument("--nhid", default=1200, type=int, help="")
parser.add_argument("--parameter", default="", type=str, help="")
parser.add_argument("--rbuff_size", default=0.05, type=float, help="")
parser.add_argument("--pseudocoreset", default=False, type=bool, help="")

# UCB HYPER-PARAMETERS
parser.add_argument(
    "--samples", default="10", type=int, help="Number of Monte Carlo samples"
)
parser.add_argument("--rho", default="-3", type=float, help="Initial rho")
parser.add_argument(
    "--sig1",
    default="0.0",
    type=float,
    help="STD foor the 1st prior pdf in scaled mixture Gaussian",
)
parser.add_argument(
    "--sig2",
    default="6.0",
    type=float,
    help="STD foor the 2nd prior pdf in scaled mixture Gaussian",
)
parser.add_argument(
    "--pi", default="0.25", type=float, help="weighting factor for prior"
)
parser.add_argument(
    "--arch", default="mlp", type=str, help="Bayesian Neural Network architecture"
)

parser.add_argument("--resume", default="no", type=str, help="resume?")
parser.add_argument("--sti", default=0, type=int, help="starting task?")

args = parser.parse_args()
utils.print_arguments(args)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


print("Using device:", args.device)
checkpoint = utils.make_directories(args)
args.checkpoint = checkpoint
print()

# Args -- Experiment
if args.experiment == "mnist2":
    from dataloaders import mnist2 as dataloader
elif args.experiment == "mnist5":
    from dataloaders import mnist5 as dataloader
elif args.experiment == "pmnist":
    from dataloaders import pmnist as dataloader
elif args.experiment == "cifar":
    from dataloaders import cifar as dataloader
elif args.experiment == "mixture":
    from dataloaders import mixture as dataloader

# Args -- Approach
if args.approach == "ucb_adaptive_lr":
    from approaches import ucb_adaptive_lr as approach
elif args.approach == "ucb_bcl":
    from approaches import ucb_bcl as approach
elif args.approach == "ucb_uniform_full":
    from approaches import ucb_uniform_full as approach
elif args.approach == "ucb_simplified_coresets":
    from approaches import ucb_simplified_coresets as approach
elif args.approach == "ucb_uniform":
    from approaches import ucb_uniform as approach
elif args.approach == "ucb_bpc":
    from approaches import ucb_bpc as approach

# Args -- Network
if (
    args.experiment == "mnist2"
    or args.experiment == "pmnist"
    or args.experiment == "mnist5"
):
    from networks import mlp_ucb as network
else:
    from networks import resnet_ucb as network


########################################################################################################################
print()
print("Starting this run on :")
print(datetime.now().strftime("%Y-%m-%d %H:%M"))

# Load
print("Load data...")
data, taskcla, inputsize = dataloader.get(data_path=args.data_path, seed=args.seed)
print("Input size =", inputsize, "\nTask info =", taskcla)
args.num_tasks = len(taskcla)
args.inputsize, args.taskcla = inputsize, taskcla

shared_model_task_cache["args"] = args
# Inits
print("Inits...")
model = network.Net(args).to(args.device)


print("-" * 100)
appr = approach.Appr(model, args=args)
print("-" * 100)

# args.output=os.path.join(args.results_path, datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
print("-" * 100)

if args.resume == "yes":
    checkpoint = torch.load(
        os.path.join(args.checkpoint, "model_{}.pth.tar".format(args.sti))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device=args.device)
else:
    args.sti = 0


# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

for t, ncla in taskcla[args.sti :]:
    print("*" * 100)
    print("Task {:2d} ({:s})".format(t, data[t]["name"]))
    print("*" * 100)

    # Get data
    xtrain = data[t]["train"]["x"].to(args.device)
    ytrain = data[t]["train"]["y"].to(args.device)
    xvalid = data[t]["valid"]["x"].to(args.device)
    yvalid = data[t]["valid"]["y"].to(args.device)
    task = t

    # Train
    appr.train(task, xtrain, ytrain, xvalid, yvalid)
    shared_model_task_cache["prev_task_data"][t] = {
        "xtrain": xtrain,
        "ytrain": ytrain,
        "xvalid": xvalid,
        "yvalid": yvalid,
        "ncla": ncla,
    }
    print("-" * 100)

    # Test
    for u in range(t + 1):
        xtest = data[u]["test"]["x"].to(args.device)
        ytest = data[u]["test"]["y"].to(args.device)
        test_loss, test_acc = appr.eval(u, xtest, ytest)
        print(
            ">>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}% <<<".format(
                u, data[u]["name"], test_loss, 100 * test_acc
            )
        )
        acc[t, u] = test_acc
        lss[t, u] = test_loss

    # Save
    print("Save at " + args.checkpoint)
    np.savetxt(
        os.path.join(
            args.checkpoint,
            "{}_{}_{}.txt".format(args.experiment, args.approach, args.seed),
        ),
        acc,
        "%.5f",
    )


utils.print_log_acc_bwt(args, acc, lss)
print("[Elapsed time = {:.1f} h]".format((time.time() - tstart) / (60 * 60)))
