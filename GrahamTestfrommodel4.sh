#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --mem=64000M
#SBATCH --job-name RL_TrainAndTest
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --time=6-23:00:00
#SBATCH --account=rrg-khisti

module load python/3.8.10
source RLVirtualEnv/bin/activate
python3 ./TestFromModels_InOrder4.py