#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name RL_TrainAndTest
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --time=03:00:00

module module load anaconda3/2021.05
source activate pytorch_env
python3 ./TrainAndTestGradientQL.py