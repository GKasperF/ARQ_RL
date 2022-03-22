from joblib import Parallel, delayed
import multiprocessing
from LowerBound.BruteForceUtilityFunctions import ProbabilitySchedulingMarkovChannel_dec
import pickle
import numpy as np

#Set Parameters
T = 20
Tf = 10
P_matrix = np.array([[0.99, 0.01, 0.0, 0.0], [0.0, 0.7, 0.3, 0.0], [0.0, 0.0, 0.7, 0.3], [0.3, 0.0, 0.0, 0.7]])
epsilon_vector = np.array([[0.05], [1], [1], [1]])
h = 1

num_cores = multiprocessing.cpu_count()
store_results = Parallel(n_jobs = num_cores)(delayed(ProbabilitySchedulingMarkovChannel_dec)(P_matrix, epsilon_vector, Tf, T, i) for i in range(1,2**(T+1)) )

with open('Data/BruteForceFritchmanFull.pickle', 'wb') as f:
    pickle.dump(store_results, f)