#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --tasks-per-node=2
#SBATCH --mem=64000M
#SBATCH --job-name RL_TrainAndTest
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --time=5-23:00:00
#SBATCH --account=rrg-khisti

module load python/3.8.10
source RLVirtualEnv/bin/activate
python3 ./TrainAndTestGradientQL_RNN2.py