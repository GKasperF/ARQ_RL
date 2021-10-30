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

def TrainAndTest(alpha_reward, beta_reward, Tf, Nit, discount_factor, num_episodes, epsilon, batch, Channel):
    Channel_Local = copy.deepcopy(Channel)
    TransEnv = Envs.EnvFeedbackGeneral(Tf, alpha_reward, beta_reward, Channel_Local, batch)
    
    Q, policy = Train(TransEnv, discount_factor, num_episodes, epsilon)
    
    average_reward, average_transmissions, average_recovery = Test(TransEnv, policy, Nit)
    
    return(average_reward, average_transmissions, average_recovery)

def Train(env, discount_factor, num_episodes, epsilon):
    Qfunction = QL.QApproxFunction(env.observation_space.n, env.action_space.n)
    for i in range(len(num_episodes)):
        Q, policy = QL.GradientQLearning(env, num_episodes[i], Qfunction, discount_factor, epsilon[i])
    
    return(Qfunction, policy)

def Test(env, policy, Nit):
    reward_save = np.zeros((Nit, 4))
    for i in range(Nit):
        done = 0
        state = env.reset()
        reward_acc = 0
        transmissions = 0
        time_instant = 0
        number_successes = 0
        while 1:
          action_probabilities = policy(state)
          action_index = np.random.choice(np.arange(
                    len(action_probabilities)),
                      p = action_probabilities)

          # take action and get reward, transit to next state
          if action_index == 0:
            transmissions += 1

          next_state, reward, done, SuccessF = env.step(env.actions[action_index])


          # Update statistics
          reward_acc += reward
          time_instant += 1
          state = next_state
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


#Channel = Envs.GilbertElliott(0.25, 0.25, 0, 1)
Channel = Envs.iidchannel(0.25)
#num_cores = multiprocessing.cpu_count()
num_cores = 1
#alpha_range = np.arange(1.4, 1.71, 0.1)
alpha_range = [1.4]
beta_reward = 5
Tf = 10
Nit = 100000
discount_factor = 0.95
#num_episodes = [20000, 20000, 100000, 200000, 500000]
#epsilon = [0.8, 0.6, 0.3, 0.2, 0.1]

num_episodes = [1000]
epsilon = [0.5]

batches = 1

store_results = Parallel(n_jobs = num_cores)(delayed(TrainAndTest)(alpha_reward, beta_reward, Tf, Nit, discount_factor, num_episodes, epsilon, batches, Channel) for alpha_reward in alpha_range)

with open('Data/AgentNNRLresults.pickle', 'wb') as f:
    pickle.dump(store_results, f)

