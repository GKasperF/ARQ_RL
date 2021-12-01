#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name RL_TrainModel
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --time=22:00:00
#SBATCH --gpus-per-node=1

module load anaconda3/2021.05
source activate pytorch_env
python3 ./TrainAndSaveModelMC.py
