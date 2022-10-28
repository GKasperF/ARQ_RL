#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name TestFromModels
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1

module load anaconda3/2021.05
source activate pytorch_env
python3 ./TestFromModels_Multi3.py