## Prerequisites:
- Linux-64
- Conda

## Environment Setup

Run the following commands one after the other:

```
conda create --name ucb python=3.6
conda activate ucb
pip install -r requirements.txt
```

## Generating Results

### Experiments:

The following expreimental tasks (`--experiment`) can be chosen from:
* MNIST2: `mnist2`
* MNIST5: `mnist5`
* CIFAR: `cifar`
* PMNIST: `pmnist`

### Approaches:

The following approaches (`--approach`) are present in the repositories. For other baselines, please refer to the code present in their corresponding repositories. 

* Adaptive Learning Rate (`ucb_adaptive_lr`) : This refers to the main approach presented in the paper [Uncertainty-guided Continual Learning with Bayesian Neural Networks](https://arxiv.org/abs/1906.02425).
* Bayesian Continual Learning (`ucb_bcl`) : For this, we disable the adaptive learning rate of the previous approach and implement Bayesian Continual Learning (BCL).
* Uniform Sampling per Epoch (`ucb_uniform_full`) : We build on top of the BCL approach and uniformly sample a set of inputs from a randomly selected task from one of the previous tasks at each epoch. The size of the sample can be controlled by adjusting `--rbuff_size` parameter of `run.py`. This is a percentage value, expressed as a `float`.
* Simplified Bayesian Coresets (`ucb_simplified_coresets`) : We build on top of the BCL approach and select several batches of the current task inputs and minimize the reverse KL-divergence between posterior of the data sample and the entire dataset. The batch obtained this way serves as the coreset for training of subsequent tasks.
* Uniform Sampling Coresets (`ucb_uniform`) : We build on top of the BCL approach and select a batch of images uniformly from inputs of the current task. The batch obtained serves as the coreset for training of subsequent tasks.
* Uniform Bayesian Coresets and Pseudocoresets (`ucb_bpc`) : We build on top of the BCL approach and generate Bayesian Coresets or Pseudocoresets (`--pseudocoreset True`) for training of subsequent tasks.

### Example Commands:

In general to run any of the experiments, the following command can be used. Any additional parameters that need to be tweaked, can be done by supplying them via the CLI. Have a look in [run.py](src/run.py) to see what all customizations are available.

`python src/run.py --experiment <EXPERIMENT_NAME> --approach <APPROACH_NAME>`

#### Running `mnist2` with `ucb_adaptive_lr` for `20` epochs

`python src/run.py --experiment mnist2 --approach ucb_adaptive_lr --nepochs 20`

#### Running `mnist5` with `ucb_bpc` for `5` epochs using Bayesian Pseudo Coresets

`python src/run.py --experiment mnist5 --approach ucb_bpc --nepochs 5 --pseudocoreset True`

#### Running `pmnist` with `ucb_bpc` for `10` epochs using Bayesian  Coresets

`python src/run.py --experiment pmnist --approach ucb_bpc --nepochs 10`

#### Running `cifar` with `ucb_uniform_full` for `20` epochs using Bayesian  Coresets

`python src/run.py --experiment cifar --approach ucb_uniform_full --nepochs 20`

## License
This source code is released under The MIT License found in the [LICENSE](./LICENSE) file in the root directory of this source tree.

## Acknowledgements
Our code structure is inspired by [UCB](https://github.com/SaynaEbrahimi/UCB).