from joblib import Parallel, delayed
import multiprocessing
from LowerBound.BruteForceUtilityFunctions import ProbabilitySchedulingGE_full_dec
import pickle

#Set Parameters
T = 20
Tf = 10
alpha = 0.01
beta = 0.25
epsilon = 0.05
h = 1

#ProbabilitySchedulingGE_full_dec(alpha, beta, epsilon, h, Tf, T, 3)

num_cores = multiprocessing.cpu_count()
store_results = Parallel(n_jobs = num_cores)(delayed(ProbabilitySchedulingGE_full_dec)(alpha, beta, epsilon, h, Tf, T, i) for i in range(1,2**(T+1)) )

with open('Data/BruteForceGEFull.pickle', 'wb') as f:
    pickle.dump(store_results, f)