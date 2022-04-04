#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --tasks-per-node=2
#SBATCH --job-name RL_TrainAndTest
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --time=23:00:00
#SBATCH --account=rrg-khisti

module load python/3.8.10
source /scratch/kasperf/ARQ_RL/RLVirtualEnv/bin/activate
pip3 install numpy
pip3 install joblib
pip3 install gym
pip3 install --no-index torch torchvision torchtext torchaudio

python3 ./TrainAndTestGradientQL_RNN.py