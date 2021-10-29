import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed
import multiprocessing


from BruteForceUtilityFunctions import ProbabilityScheduling, ProbabilityScheduling_dec, lower_convex_hull

T = 10
Tf = 4
epsilon = 0.25

num_cores = 5
store_results = Parallel(n_jobs = num_cores)(delayed(ProbabilityScheduling_dec)(epsilon, Tf, T, i) for i in range(1,2**(T+1)))

with open('Data/BruteForceIid.pickle', 'wb') as f:
    pickle.dump(store_results, f)