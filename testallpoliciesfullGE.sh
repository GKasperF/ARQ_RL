#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name RL_TrainAndTest
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --time=06:00:00

module load python/3.8.5
source RLVirtualEnv/bin/activate
python ./TestAllPoliciesFullGE.py
