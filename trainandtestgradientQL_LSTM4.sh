#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name RL_TrainAndTest
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --time=23:00:00
#SBATCH --gpus-per-node=4
#SBATCH --partition=compute_full_node

module load anaconda3/2021.05
source activate pytorch_env
python3 ./TrainAndTestGradientQL_LSTM4.py