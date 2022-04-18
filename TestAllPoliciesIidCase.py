import pickle
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

from LowerBound.BruteForceUtilityFunctions import ProbabilityScheduling_dec
# Set Parameters
T = 20
Tf = 10
epsilon = 0.1
num_cores = multiprocessing.cpu_count()
# Compute all possible transmission schedules
store_results = Parallel(n_jobs = num_cores)(delayed(ProbabilityScheduling_dec)(epsilon, Tf, T, i) for i in tqdm(range(1,2**(T+1))))
# Another lower bound: number of transmissions >= 1 / (1 - epsilon). 
delta = 0.05 #Allow some delta gap
i=0

with open('Data/BruteForceIid_Example.pickle', 'wb') as f:
    pickle.dump(store_results, f)
