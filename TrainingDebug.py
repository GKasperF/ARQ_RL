import numpy as np
from time import time
import gym
from gym import error, spaces, utils
import copy
from gym.utils import seeding
from joblib import Parallel, delayed
import multiprocessing
import pickle
import ReinforcementLearning.QlearningFunctions as QL
import Envs.PytorchEnvironments as Envs
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#device = 'cpu'

def TrainDebug(env, discount_factor, num_episodes, epsilon):
    Qfunction = QL.QApproxFunction(env.observation_space.n, env.action_space.n).to(device)
    for i in range(len(num_episodes)):
        Qfunction, policy, Debug = QL.GradientQLearningDebug(env, num_episodes[i], Qfunction, discount_factor, epsilon[i])
    
    return(Qfunction, policy, Debug)

num_cores = 2
Channel = Envs.GilbertElliott(0.25, 0.25, 0, 1)
TransEnv = Envs.EnvFeedbackGeneral(10, 1.4, 5, Channel, 1)
TransEnv = TransEnv.to(device)


t0 = time()
Q, policy, Debug = TrainDebug(TransEnv, 0.95, [10000], [0.5])
t1 = time()

print('Training takes {} seconds'.format(t1-t0))

with open('Data/SaveModel.pickle', 'wb') as f:
	pickle.dump(Q, f)

with open('Data/SaveDebug.pickle', 'wb') as f:
	pickle.dump(Debug, f)




#store_results = Parallel(n_jobs = num_cores)(delayed(TrainAndTest)(alpha_reward, beta_reward, Tf, Nit, discount_factor, num_episodes, epsilon, batches, Channel) for alpha_reward in alpha_range)

#Q, policy = Train(env, 0.95, [1000], [0.5])
