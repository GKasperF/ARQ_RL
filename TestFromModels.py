import numpy as np
from time import time
from joblib import Parallel, delayed
import multiprocessing
import pickle
import Envs.PytorchEnvironments as Envs
import Envs.Environments as EnvsTable
import torch
import copy
import dill
import io
import glob

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def Test(env, Q, Nit, batch=1):
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


batch = 1000
Channel = Envs.GilbertElliott(0.25, 0.25, 0, 1, batch).to(device)
TransEnv = Envs.EnvFeedbackGeneral(10, 0.6, 5, Channel, batch, M = 1)
TransEnv = TransEnv.to(device)


class CPU_Unpickler(pickle.Unpickler):
  def find_class(self, module, name):
      if module == 'torch.storage' and name == '_load_from_bytes':
          return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
      else: return super().find_class(module, name)


all_results = []

path = 'Data/ModelCNNFromDataset_Memory*.pickle'
for filename in glob.glob(path):
  with open(filename, 'rb') as f:
    Q = CPU_Unpickler(f).load()
  Q = Q.to(device)
  t0 = time()
  print(filename)
  results = Test(TransEnv, Q, 100000, batch)
  t1 = time()

  print('Testing takes {} seconds'.format(t1-t0))
  all_results.append(results)

with open('Data/AgentCNNRLresultsTestBatch_FromDataset_Memory1.pickle', 'wb') as f:
  pickle.dump(all_results, f)