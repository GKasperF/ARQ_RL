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

def Train(env, discount_factor, num_episodes, epsilon):
    Qfunction = QL.QApproxFunction(env.observation_space.n, env.action_space.n).to(device)
    for i in range(len(num_episodes)):
        Qfunction, policy = QL.GradientQLearning(env, num_episodes[i], Qfunction, discount_factor, epsilon[i])
    
    return(Qfunction, policy)

Channel = Envs.iidchannel(0.25)
env = Envs.EnvFeedbackGeneral(4, 1.4, 5, Channel, 1)
env = env.to(device)
t_begin = time()
Q, policy = Train(env, 0.95, [1000], [0.5])
t_end = time()
print('Training takes {} seconds'.format(t_end - t_begin))