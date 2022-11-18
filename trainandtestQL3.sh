#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --mem=64000M
#SBATCH --job-name RL_TrainAndTest_QTable_GE_Isolated
#SBATCH --mail-user=gustavo.kasperfacenda@mail.utoronto.ca
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --time=20:00:00
#SBATCH --account=rrg-khisti

module load python/3.8.10
source RLVirtualEnv/bin/activate
python3 ./TrainAndTestQL3.py
