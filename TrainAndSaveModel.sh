#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name RL_TrainModel
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1

module load anaconda3/2021.05
source activate pytorch_env
python3 ./TrainAndSaveModel.py SaveModelAlpha14.pickle 1.4
python3 ./TrainAndSaveModel.py SaveModelAlpha01.pickle 0.1
python3 ./TrainAndSaveModel.py SaveModelAlpha10.pickle 1.0


