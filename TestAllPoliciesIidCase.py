import pickle
from joblib import Parallel, delayed
import multiprocessing

from LowerBound.BruteForceUtilityFunctions import ProbabilityScheduling_dec
# Set Parameters
T = 10
Tf = 4
epsilon = 0.25
num_cores = multiprocessing.cpu_count()
# Compute all possible transmission schedules
store_results = Parallel(n_jobs = num_cores)(delayed(ProbabilityScheduling_dec)(epsilon, Tf, T, i) for i in range(1,2**(T+1)))
# Another lower bound: number of transmissions >= 1 / (1 - epsilon). 
delta = 0.05 #Allow some delta gap
i=0
while 1:
    point = store_results[i]
    if point[0] < (1/(1 - epsilon) - delta):
        store_results.pop(i)
    else:
        i+=1
    
    if i == len(store_results):
        break

with open('Data/BruteForceIid.pickle', 'wb') as f:
    pickle.dump(store_results, f)
