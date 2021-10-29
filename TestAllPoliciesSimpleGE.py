from joblib import Parallel, delayed
import multiprocessing
from LowerBound.BruteForceUtilityFunctions import ProbabilitySchedulingGE_Simple, ProbabilitySchedulingGE_simple_dec, lower_convex_hull, convex_hull
import pickle

#Set Parameters
T = 20
Tf = 10
alpha = 0.25
beta = 0.25

num_cores = multiprocessing.cpu_count()
store_results = Parallel(n_jobs = num_cores)(delayed(ProbabilitySchedulingGE_simple_dec)(alpha, beta, Tf, T, i) for i in range(1,2**(T+1)) )

with open('Data/BruteForceGESimple.pickle', 'wb') as f:
    pickle.dump(store_results, f)
