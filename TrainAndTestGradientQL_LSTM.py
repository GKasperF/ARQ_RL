import numpy as np
import time
import copy
from joblib import Parallel, delayed
import multiprocessing
import pickle
import ReinforcementLearning.QlearningFunctions as QL
import Envs.PytorchEnvironments as Envs
import torch
import sys
import os

deadline = float(90)

q = []
if torch.cuda.is_available():
  num_cores = torch.cuda.device_count()
  for i in range(num_cores):
    q.append('cuda:'+'{}'.format(i))
else:
  num_cores = multiprocessing.cpu_count()
  for i in range(num_cores):
    q.append('cpu')

test_file = 'Data/AgentCNN_LSTM_DRQN_LongerDeadline_RLresultsTestBatch_Fritchman.pickle'
if os.path.isfile(test_file):
  with open(test_file, 'rb') as f:
    results_dict = pickle.load(f)
else:
  results_dict = {}

def TrainAndTest(alpha_reward, beta_reward, Tf, Nit, discount_factor, num_episodes, epsilon, batch, Channel):
    device = q.pop()
    Channel_Local = copy.deepcopy(Channel).to(device)
    string_alpha = str(alpha_reward.numpy())
    alpha_reward = alpha_reward.to(device)
    TransEnv = Envs.EnvFeedbackGeneral(Tf, alpha_reward, beta_reward, Channel_Local, batch, M=5)
    TransEnv = TransEnv.to(device)
    model_file = 'ModelCNN_LSTM_DRQN_Batch_Fritchman'+string_alpha+'.pickle'
    if os.path.isfile('Data/'+model_file):
      with open('Data/'+model_file, 'rb') as f:
        Q = pickle.load(f)
        Q = Q.to(device)
    else:
      t0 = time.time()
      Q, policy = Train(TransEnv, discount_factor, num_episodes, epsilon)
      t1 = time.time()
      with open('Data/'+model_file, 'wb') as f:
        pickle.dump(Q, f)
      print('Training takes {} seconds'.format(t1 - t0))

    

    if model_file not in results_dict:
      print('Start testing for alpha {}'.format(alpha_reward))
      t0 = time.time()
      result = Test(TransEnv, Q, Nit, batch)
      t1 = time.time()
      print('Testing for alpha {} takes {} seconds'.format(alpha_reward, t1 - t0))
      results_dict[model_file] = result
    else:
      result = results_dict[model_file]


    q.append(device)

    with open(test_file, 'wb') as f:
      pickle.dump(results_dict, f)

    return(result)

def Train(env, discount_factor, num_episodes, epsilon):
    Qfunction = QL.QApproxFunction_LSTM(env.observation_space.n, env.action_space.n, 1000).to(env.device)
    #Qfunction = QL.QApproxFunction(env.observation_space.n, env.action_space.n, 1000).to(env.device)
    lr_list = [0.001, 0.001, 0.001, 0.0001, 0.00001]
    for i in range(len(num_episodes)):
        Qfunction, policy, _ = QL.GradientQLearningLSTM(env, num_episodes[i], Qfunction, discount_factor, epsilon[i], UpdateEpisodes=10, UpdateTargetEpisodes= 100, lr=lr_list[i])
        #Qfunction, policy, _ = QL.GradientQLearningDebug(env, num_episodes[i], Qfunction, discount_factor, epsilon[i], UpdateEpisodes= 10, UpdateTargetEpisodes= 100, lr = lr_list[i])
    
    return(Qfunction, policy)

def Test(env, Q, Nit, batch):
    device = env.device
    reward_acc = torch.zeros(batch).to(device)
    transmissions = torch.zeros(batch).to(device)
    time_instant = torch.zeros(batch).to(device)
    number_successes = torch.zeros(batch).to(device)

    torch_ones = torch.ones(batch).to(device).type(torch.int64)

    reward_save = torch.empty((0, 4)).to(device)
    for i in range(int(Nit/batch)):
        done = 0
        reward_acc[:] = 0
        transmissions[:] = 0
        time_instant[:] = 1
        number_successes[:] = 0
        state = env.reset()
        h_in = torch.zeros((5, batch, Q.hidden_size_LSTM)).to(device)
        c_in = torch.zeros((5, batch, Q.hidden_size_LSTM)).to(device)
        SuccessF = torch.zeros(batch).to(device)
        while 1:
          Q_values, (h_out, c_out) = Q(state, h_in, c_in)
          action_index_temp = torch.argmax(Q_values, dim = 1)

          #Force transmissions if past deadline:
      
          action_index = torch.where(time_instant > deadline, torch_ones, action_index_temp)

          # take action and get reward, transit to next state
          transmissions[torch.logical_not(SuccessF)] = transmissions[torch.logical_not(SuccessF)] + action_index.reshape(len(action_index))[torch.logical_not(SuccessF)]

          next_state, reward, done, SuccessF = env.step(action_index)


          # Update statistics
          reward_acc += reward.reshape(len(reward))
          time_instant[ torch.logical_not(SuccessF)] += 1
          state = next_state
          h_in = h_out.detach() 
          c_in = c_out.detach()
          # if torch.any(time_instant > env.Tf) and torch.any(transmissions == 0):
          #   print('Learned bad policy')
          #   break
          if done:
            break
        
        temp = torch.cat( (reward_acc.reshape(batch, 1), transmissions.reshape(batch, 1), time_instant.reshape(batch, 1), number_successes.reshape(batch, 1)), dim = 1)
        reward_save = torch.cat( (reward_save, copy.deepcopy(temp)), dim = 0)
        
    average_reward = torch.mean(reward_save[:, 0])
    average_transmissions = torch.mean(reward_save[:, 1])
    average_recovery = torch.mean(reward_save[:, 2]) - env.Tf
    
    return(average_reward, average_transmissions, average_recovery)


alpha_range = torch.arange(0.1, 5.5, 0.1)
#alpha_range = torch.tensor([0.1, 0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0])
beta_reward = 5
Tf = 10
Nit = 10000
epsilon = [0.8, 0.6, 0.3, 0.2, 0.1]
discount_factor = 0.95

batches = 100

#Channel = Envs.GilbertElliott(0.1, 0.25, 0.05, 1, batch_size).to(device)
#Channel = Envs.GilbertElliott(0.25, 0.25, 0.0, 1, batch_size).to(device)
Channel = Envs.Fritchman(0.1, 0.5, 0.05, 4, batches)
#Channel = Envs.iidchannel(0.1, batches)

num_episodes = [int(2000), int(2000), int(10000), int(20000), int(50000)]

store_results = Parallel(n_jobs = num_cores, require='sharedmem')(delayed(TrainAndTest)(alpha_reward, beta_reward, Tf, Nit, discount_factor, num_episodes, epsilon, batches, Channel) for alpha_reward in alpha_range)
#store_results = TrainAndTest(alpha_range[0], beta_reward, Tf, Nit, discount_factor, num_episodes, epsilon, batches, Channel)
with open('Data/AgentCNN_LSTM_DRQN_RL_All_resultsTestBatch_Fritchman.pickle', 'wb') as f:
    pickle.dump(store_results, f)