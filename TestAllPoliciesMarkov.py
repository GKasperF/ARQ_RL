from joblib import Parallel, delayed
import multiprocessing
from LowerBound.BruteForceUtilityFunctions import ProbabilitySchedulingMarkovChannel_dec
import pickle
import numpy as np
from tqdm import tqdm

#Set Parameters
T = 20
Tf = 10
P_matrix = np.array([[0.9, 0.1, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 0.0, 0.5]])
epsilon_vector = np.array([[0.05], [1], [1], [1]])

# P_matrix = np.array([[0.75, 0.25], [0.25, 0.75]])
# epsilon_vector = np.array([[0.0], [1]])

#Channel_Local = Envs.GilbertElliott(0.25, 0.25, 0.0, 1)
#Channel_Local = Envs.Fritchman(0.1, 0.5, 0.05, 4)

num_cores = multiprocessing.cpu_count()
store_results = Parallel(n_jobs = num_cores)(delayed(ProbabilitySchedulingMarkovChannel_dec)(P_matrix, epsilon_vector, Tf, T, i) for i in tqdm(range(1,2**(T+1))) )

with open('Data/BruteForce_Fritchman_Example.pickle', 'wb') as f:
    pickle.dump(store_results, f)