import numpy as np
import LowerBound.BruteForceUtilityFunctions as BF

alpha = 0.1
beta = 0.3

epsilon_vector = np.array([[0.01], [1.0]])

P_matrix = np.array([[1 - alpha, alpha], [beta, 1 - beta]]) #define the transition matrix. When we have no new information, update is p0 = p0 * P_matrix

Tf = 10
T = 20

S_dec = 2**21 - 1

test = BF.ProbabilitySchedulingMarkovChannel_dec(P_matrix, epsilon_vector, Tf, T, S_dec)