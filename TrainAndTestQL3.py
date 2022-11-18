import numpy as np
import time
import copy
from joblib import Parallel, delayed
import multiprocessing
import pickle
import ReinforcementLearning.QlearningTable as QL
import Envs.Environments as Envs
from collections import defaultdict
import os

test_file = 'Data/Agent_QTable_results_GE_Isolated_Example_Dict.pickle'
if os.path.isfile(test_file):
  with open(test_file, 'rb') as f:
    results_dict = pickle.load(f)
else:
  results_dict = {}

def TrainAndTest(alpha_reward, beta_reward, Tf, Nit, discount_factor, num_episodes, alpha, epsilon, Channel, M):
    Channel_Local = copy.deepcopy(Channel)
    TransEnv = Envs.EnvFeedbackGeneral(Tf, alpha_reward, beta_reward, Channel_Local, M)
    string_alpha = str(alpha_reward)
    model_file = 'Model_GE_Isolated_Example_QTable'+string_alpha+'.pickle'
    if os.path.isfile('Data/'+model_file):
      with open('Data/'+ model_file, 'rb') as f:
        policy = pickle.load(f)
    else:
      t0 = time.time()
      Q, policy = Train(TransEnv, discount_factor, num_episodes, alpha, epsilon)
      t1 = time.time()
      print('Training takes {} seconds'.format(t1 - t0))
      with open('Data/'+model_file, 'wb') as f:
        pickle.dump(policy, f)

    
    
    average_reward, average_transmissions, average_recovery = Test(TransEnv, policy, Nit)
    
    return(average_reward, average_transmissions, average_recovery)

def Train(env, discount_factor, num_episodes, alpha, epsilon):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for i in range(len(num_episodes)):
        Q, policy = QL.qLearning(env, num_episodes[i], Q, discount_factor, alpha[i], epsilon[i])
    
    return(Q, policy)

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


Channel = Envs.GilbertElliott(0.1, 0.25, 0.05, 1)
#Channel = Envs.iidchannel(0.1)
num_cores = multiprocessing.cpu_count()
alpha_range = np.arange(0.1, 5.5, 0.1)
beta_reward = 5
Tf = 10
Nit = 500000
discount_factor = 0.95
#num_episodes = [100000, 100000, 500000, 1000000, 2500000]
num_episodes = [2000, 2000, 10000, 20000, 50000]
M = 1
epsilon = [0.8, 0.6, 0.3, 0.2, 0.1]
alpha = [0.5, 0.2, 0.01, 0.001, 0.0001]
store_results = Parallel(n_jobs = num_cores)(delayed(TrainAndTest)(alpha_reward, beta_reward, Tf, Nit, discount_factor, num_episodes, alpha, epsilon, Channel, M) for alpha_reward in alpha_range)

with open('Data/AgentRLresults_QTable_GE_Isolated_Example.pickle', 'wb') as f:
    pickle.dump(store_results, f)

