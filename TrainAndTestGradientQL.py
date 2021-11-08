import numpy as np
import time
import gym
from gym import error, spaces, utils
import copy
from gym.utils import seeding
from joblib import Parallel, delayed
import multiprocessing
import pickle
import ReinforcementLearning.QlearningFunctions as QL
import Envs.PytorchEnvironments as Envs
from collections import defaultdict
import torch
import random

q = []
if torch.cuda.is_available():
  num_cores = torch.cuda.device_count()
  for i in range(num_cores):
    q.append('cuda:'+'{}'.format(i))
else:
  num_cores = multiprocessing.cpu_count()
  for i in range(num_cores):
    q.append('cpu')

def TrainAndTest(alpha_reward, beta_reward, Tf, Nit, discount_factor, num_episodes, epsilon, batch, Channel):
    device = q.pop()
    Channel_Local = copy.deepcopy(Channel)
    alpha_reward = alpha_reward.to(device)
    TransEnv = Envs.EnvFeedbackGeneral(Tf, alpha_reward, beta_reward, Channel_Local, batch)
    TransEnv = TransEnv.to(device)
    
    Q, policy = Train(TransEnv, discount_factor, num_episodes, epsilon)

    result = Test(TransEnv, Q, Nit)
    q.append(device)

    return(result)

def Train(env, discount_factor, num_episodes, epsilon):
    Qfunction = QL.QApproxFunction(env.observation_space.n, env.action_space.n, 1000).to(env.device)
    for i in range(len(num_episodes)):
        Q, policy = QL.GradientQLearning(env, num_episodes[i], Qfunction, discount_factor, epsilon[i])
    
    return(Qfunction, policy)

def Test(env, Q, Nit):
    reward_save = np.zeros((Nit, 4))
    for i in range(Nit):
        done = 0
        state = env.reset()
        reward_acc = 0
        transmissions = 0
        time_instant = 0
        number_successes = 0
        while 1:
          action_index = torch.argmax(Q(state), dim = 1)

          # take action and get reward, transit to next state
          if action_index == 0:
            transmissions += 1

          next_state, reward, done, SuccessF = env.step(env.actions[action_index])


          # Update statistics
          reward_acc += reward
          time_instant += 1
          state = next_state
          if time_instant > env.Tf and transmissions == 0:
            print('Learned bad policy')
            break
          if done:
            if SuccessF:
              number_successes += 1
            break
        reward_save[i][0] = reward_acc
        reward_save[i][1] = transmissions
        reward_save[i][2] = time_instant
        reward_save[i][3] = number_successes
        
    average_reward = np.mean(reward_save[:, 0])
    average_transmissions = np.mean(reward_save[:, 1])
    average_recovery = np.mean(reward_save[:, 2]) - env.Tf
    
    return(average_reward, average_transmissions, average_recovery)

Channel = Envs.GilbertElliott(0.25, 0.25, 0, 1)
alpha_range = torch.arange(0.1, 5.5, 0.1)
beta_reward = 5
Tf = 10
Nit = 10000
discount_factor = 0.95
num_episodes = [1000]
epsilon = [0.5]

batches = 1

store_results = Parallel(n_jobs = num_cores, require='sharedmem')(delayed(TrainAndTest)(alpha_reward, beta_reward, Tf, Nit, discount_factor, num_episodes, epsilon, batches, Channel) for alpha_reward in alpha_range)

with open('Data/AgentNNRLresults.pickle', 'wb') as f:
    pickle.dump(store_results, f)

