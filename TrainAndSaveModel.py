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

def TrainDebugNN(env, discount_factor, num_episodes, epsilon):
    Qfunction = QL.QApproxFunction(env.observation_space.n, env.action_space.n).to(device)
    Debug = []
    for i in range(len(num_episodes)):
        Qfunction, policy, Debug = QL.GradientQLearningDebug(env, num_episodes[i], Qfunction, discount_factor, epsilon[i])
    
    return(Qfunction, policy, Debug)

num_cores = 2
Channel = EnvsNN.GilbertElliott(0.25, 0.25, 0, 1)
TransEnv = EnvsNN.EnvFeedbackGeneral(10, 1.4, 5, Channel, 1)
TransEnv = TransEnv.to(device)

num_episodes = [20000, 20000, 100000, 200000, 500000]
epsilon = [0.8, 0.6, 0.3, 0.2, 0.1]
discount_factor = 0.95

t0 = time()
QNN, policy, Debug = TrainDebugNN(TransEnv, discount_factor, num_episodes, epsilon)
t1 = time()

print('Training NN takes {} seconds'.format(t1-t0))

with open('Data/SaveModelNN.pickle', 'wb') as f:
	pickle.dump(QNN, f)

with open('Data/SaveDebugNN.pickle', 'wb') as f:
	pickle.dump(Debug, f)