#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --account=dl
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --job-name="ucb_mixture"
#SBATCH --mem-per-cpu=16384
#SBATCH --mail-type=END

source /home/ugupta/miniconda3/etc/profile.d/conda.sh
conda activate /home/ugupta/miniconda3/envs/ucb
cd /home/ugupta/deep_learning/Deep_Learning_ETH_Fall
python src/run.py --experiment mixture --approach ucb --nepochs 20 | tee /home/ugupta/deep_learning/Deep_Learning_ETH_Fall/logs/MIXTURE.log
