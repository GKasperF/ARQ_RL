import numpy as np
import time
import copy
from joblib import Parallel, delayed
import multiprocessing
import pickle
import ReinforcementLearning.QlearningFunctions as QL
import Envs.PytorchEnvironments as Envs
import torch
import os

test_file = 'Data/AgentCNNRLresults_Fritchman_Example_RNN_Dict.pickle'
if os.path.isfile(test_file):
  with open(test_file, 'rb') as f:
    results_dict = pickle.load(f)
else:
  results_dict = {}

class ChannelModel(torch.nn.Module):
  def __init__(self, hidden_size, num_layers, output_size):
    super(ChannelModel, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.Layer1 = torch.nn.LSTM(input_size = 2, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
    self.FinalLayer = torch.nn.Linear(hidden_size, output_size)
    self.prob_layer = torch.nn.Sigmoid()

  def forward(self, x, h, c):
    L1, (h_out, c_out) = self.Layer1(x, (h, c))
    L2 = self.FinalLayer(L1)
    output = self.prob_layer(L2)
    #output = torch.sigmoid(L2)

    return output, (h_out, c_out)

with open('Data/Fritchman_Model_Example.pickle', 'rb') as f:
  RNN_Model = torch.load(f)

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
    Channel_Local = copy.deepcopy(Channel).to(device)
    RNN_Model_Local = copy.deepcopy(RNN_Model).to(device)
    string_alpha = str(alpha_reward.numpy())
    alpha_reward = alpha_reward.to(device)
    #TransEnv = Envs.EnvFeedbackGeneral(Tf, alpha_reward, beta_reward, Channel_Local, batch, M=5)
    TransEnv = Envs.EnvFeedbackRNN_GE(Tf, alpha_reward, beta_reward, Channel_Local, RNN_Model_Local, batch)
    TransEnv = TransEnv.to(device)
    model_file = 'ModelCNN_Fritchman_Example_RNN'+string_alpha+'.pickle'
    if os.path.isfile('Data/'+model_file):
      with open('Data/'+ model_file, 'rb') as f:
        Q = pickle.load(f).to(device)
    else:
      t0 = time.time()
      Q, policy = Train(TransEnv, discount_factor, num_episodes, epsilon)
      t1 = time.time()
      print('Training takes {} seconds'.format(t1 - t0))
    
    with open('Data/'+model_file, 'wb') as f:
      pickle.dump(Q, f)

    if model_file in results_dict:
      result = results_dict[model_file]
    else:
      t0 = time.time()
      result = Test(TransEnv, Q, Nit, batch)
      t1 = time.time()
      print('Testing takes {} seconds'.format(t1 - t0))
      results_dict[model_file] = result

    q.append(device)

    with open(test_file, 'wb') as f:
      pickle.dump(results_dict, f)

    with open('Data/AgentCNNRLresults_Fritchman_Example_RNN2.pickle', 'ab') as f:
      pickle.dump(result, f)

    return(result)

def Train(env, discount_factor, num_episodes, epsilon):
    Qfunction = QL.QApproxFunction(env.observation_space.n, env.action_space.n, 1000).to(env.device)
    lr_list = [0.001, 0.001, 0.001, 0.0001, 0.00001]
    for i in range(len(num_episodes)):
        Qfunction, policy, _ = QL.GradientQLearningDebug(env, num_episodes[i], Qfunction, discount_factor, epsilon[i], UpdateEpisodes= 10, UpdateTargetEpisodes= 100, lr = lr_list[i])
    
    return(Qfunction, policy)

def Test(env, Q, Nit, batch):
    device = env.device
    reward_acc = torch.zeros(batch).to(device)
    transmissions = torch.zeros(batch).to(device)
    time_instant = torch.zeros(batch).to(device)
    number_successes = torch.zeros(batch).to(device)

    reward_save = torch.empty((0, 4)).to(device)
    for i in range(int(Nit/batch)):
        done = 0
        reward_acc[:] = 0
        transmissions[:] = 0
        time_instant[:] = 1
        number_successes[:] = 0
        state = env.reset()
        SuccessF = torch.zeros(batch).to(device)
        while 1:
          action_index = torch.argmax(Q(state), dim = 1)
          # take action and get reward, transit to next state
          transmissions[torch.logical_not(SuccessF)] = transmissions[torch.logical_not(SuccessF)] + action_index.reshape(len(action_index))[torch.logical_not(SuccessF)]

          next_state, reward, done, SuccessF = env.step(action_index)


          # Update statistics
          reward_acc += reward.reshape(len(reward))
          time_instant[ torch.logical_not(SuccessF)] += 1
          state = next_state
          if torch.any(time_instant > env.Tf) and torch.any(transmissions == 0):
            print('Learned bad policy')
            break
          if done:
            break
        
        temp = torch.cat( (reward_acc.reshape(batch, 1), transmissions.reshape(batch, 1), time_instant.reshape(batch, 1), number_successes.reshape(batch, 1)), dim = 1)
        reward_save = torch.cat( (reward_save, copy.deepcopy(temp)), dim = 0)
        
    average_reward = torch.mean(reward_save[:, 0])
    average_transmissions = torch.mean(reward_save[:, 1])
    average_recovery = torch.mean(reward_save[:, 2]) - env.Tf

    print('Estimated expected reward is {} \n Expected reward is: {}'.format(Q(env.reset()), average_reward))
    
    return(average_reward, average_transmissions, average_recovery)


alpha_range = torch.arange(0.1, 5.5, 0.1)
#alpha_range = torch.arange(2.8, 5.5, 0.1)
#alpha_range = torch.tensor([0.1, 0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0])
beta_reward = 5
Tf = 10
Nit = 100000
epsilon = [0.8, 0.6, 0.3, 0.2, 0.1]
discount_factor = 0.95

batches = 100

#Channel = Envs.GilbertElliott(0.25, 0.25, 0, 1, batches)
#Channel = Envs.GilbertElliott(0.1, 0.25, 0.05, 1, batches)
Channel = Envs.Fritchman(0.1, 0.5, 0.05, 4, batches)
#Channel = Envs.iidchannel(0.1, batches)

num_episodes = [int(2000), int(2000), int(10000), int(20000), int(50000)]

store_results = Parallel(n_jobs = num_cores, require='sharedmem')(delayed(TrainAndTest)(alpha_reward, beta_reward, Tf, Nit, discount_factor, num_episodes, epsilon, batches, Channel) for alpha_reward in alpha_range)
#store_results = TrainAndTest(alpha_range[0], beta_reward, Tf, Nit, discount_factor, num_episodes, epsilon, batches, Channel)
with open('Data/AgentCNNRLresults_Fritchman_Example_RNN_Final2.pickle', 'wb') as f:
    pickle.dump(store_results, f)