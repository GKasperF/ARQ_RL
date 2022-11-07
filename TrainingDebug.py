import numpy as np
from time import time
from joblib import Parallel, delayed
import multiprocessing
import pickle
import dill
import ReinforcementLearning.QlearningFunctions as QL
import Envs.PytorchEnvironments as EnvsNN
import torch
import Envs.Environments as Envs
from collections import defaultdict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# def TrainDebugNNMC(env, num_episodes, epsilon):
#     Qfunction = QL.QApproxFunction(env.observation_space.n, env.action_space.n).to(device)
#     Debug = []
#     for i in range(len(num_episodes)):
#         Qfunction, policy, Debug = QL.GradientQLearningDebugMC(env, num_episodes[i], Qfunction, epsilon[i])
    
#     return(Qfunction, policy, Debug)

# num_cores = 2
# Channel = EnvsNN.GilbertElliott(0.25, 0.25, 0, 1)
# TransEnv = EnvsNN.EnvFeedbackGeneral(10, 1.4, 5, Channel, 1)
# TransEnv = TransEnv.to(device)

# num_episodes = [2000, 2000, 10000, 20000, 50000]
# epsilon = [0.8, 0.6, 0.3, 0.2, 0.1]

# t0 = time()
# QNNMC, policy, Debug = TrainDebugNNMC(TransEnv, num_episodes, epsilon)
# t1 = time()

# print('Training NNMC takes {} seconds'.format(t1-t0))

# with open('Data/SaveModelNNMC.pickle', 'wb') as f:
# 	pickle.dump(QNNMC, f)

# with open('Data/SaveDebugNNMC.pickle', 'wb') as f:
# 	pickle.dump(Debug, f)

def TrainDebugNN(env, discount_factor, num_episodes, epsilon):
    Qfunction = QL.QApproxFunction(env.observation_space.n, env.action_space.n, 1000).to(device)
    Debug = []
    for i in range(len(num_episodes)):
        Qfunction, policy, Debug = QL.GradientQLearningDebug(env, num_episodes[i], Qfunction, discount_factor, epsilon[i])
    
    return(Qfunction, policy, Debug)

num_cores = 2
Channel = EnvsNN.GilbertElliott(0.25, 0.25, 0, 1)
TransEnv = EnvsNN.EnvFeedbackGeneral(10, 1.4, 5, Channel, 1)
TransEnv = TransEnv.to(device)

num_episodes = [2000, 2000, 10000, 20000, 50000]
epsilon = [0.8, 0.6, 0.3, 0.2, 0.1]
discount_factor = 0.95

t0 = time()
QNN, policy, Debug = TrainDebugNN(TransEnv, discount_factor, num_episodes, epsilon)
t1 = time()

print('Training NN takes {} seconds'.format(t1-t0))

# with open('Data/SaveModelNN.pickle', 'wb') as f:
# 	pickle.dump(QNN, f)

# with open('Data/SaveDebugNN.pickle', 'wb') as f:
# 	pickle.dump(Debug, f)


# def TrainDebug(env, discount_factor, num_episodes, alpha, epsilon):
#     Q = defaultdict(lambda: np.zeros(env.action_space.n))
#     for i in range(len(num_episodes)):
#         Q, policy = QL.qLearning(env, num_episodes[i], Q, discount_factor, alpha[i], epsilon[i])
    
#     return(Q, policy)

# device = torch.device('cpu')

# Channel = Envs.GilbertElliott(0.25, 0.25, 0, 1)
# TransEnv = Envs.EnvFeedbackGeneral(10, 1.4, 5, Channel)

# Nit = 100000
# discount_factor = 0.95
# num_episodes = [20000, 20000, 100000, 200000, 500000]
# epsilon = [0.8, 0.6, 0.3, 0.2, 0.1]
# alpha = [0.5, 0.2, 0.01, 0.001, 0.0001]

# t0 = time()
# Q, policy = TrainDebug(TransEnv, discount_factor, num_episodes, alpha, epsilon)
# t1 = time()

# print('Training Table takes {} seconds'.format(t1-t0))

# with open('Data/SaveModelTable.pickle', 'wb') as f:
# 	dill.dump(Q, f)




# import os 
# os.system('shutdown -s -t 10')

#store_results = Parallel(n_jobs = num_cores)(delayed(TrainAndTest)(alpha_reward, beta_reward, Tf, Nit, discount_factor, num_episodes, epsilon, batches, Channel) for alpha_reward in alpha_range)

#Q, policy = Train(env, 0.95, [1000], [0.5])
