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


batch = 100
Channel = Envs.GilbertElliott(0.25, 0.25, 0, 1, batch).to(device)
TransEnv = Envs.EnvFeedbackGeneral(10, 0.6, 5, Channel, batch, M = 5)
TransEnv = TransEnv.to(device)


class CPU_Unpickler(pickle.Unpickler):
  def find_class(self, module, name):
      if module == 'torch.storage' and name == '_load_from_bytes':
          return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
      else: return super().find_class(module, name)


all_results = []

path = 'Data/ModelCNNBatch*.pickle'
for filename in glob.glob(path):
  with open(filename, 'rb') as f:
    Q = CPU_Unpickler(f).load()
  Q = Q.to(device)
  t0 = time()
  print(filename)
  results = Test(TransEnv, Q, 10000, batch)
  t1 = time()

  print('Testing takes {} seconds'.format(t1-t0))
  all_results.append(results)
  # trans = results[1]
  # delay = results[2]

  # print('Results are: \n Expected number of transmissions: {} \n Expected delay: {}'.format(trans, delay))
  # print('Expected reward is: {}'.format(results[0]))

with open('Data/AgentCNNRLresultsTestBatch_Memory1.pickle', 'wb') as f:
  pickle.dump(all_results, f)



# with open('Data/SaveModelTable.pickle', 'rb') as f:
#   QTable = dill.load(f)

# Channel = EnvsTable.GilbertElliott(0.25, 0.25, 0, 1)
# TransEnv = EnvsTable.EnvFeedbackGeneral(10, 1.4, 5, Channel)

# def Test(env, Qtable, Nit):
#     reward_save = np.zeros((Nit, 4))
#     for i in range(Nit):
#         done = 0
#         state = env.reset()
#         reward_acc = 0
#         transmissions = 0
#         time_instant = 0
#         number_successes = 0
#         while 1:
#           action_index = np.argmax( Qtable[state.tobytes()])

#           # take action and get reward, transit to next state
#           if action_index == 0:
#             transmissions += 1

#           next_state, reward, done, SuccessF = env.step(env.actions[action_index])


#           # Update statistics
#           reward_acc += reward
#           time_instant += 1
#           state = next_state
#           if done:
#             if SuccessF:
#               number_successes += 1
#             break
            
            
#         reward_save[i][0] = reward_acc
#         reward_save[i][1] = transmissions
#         reward_save[i][2] = time_instant
#         reward_save[i][3] = number_successes
        
#     average_reward = np.mean(reward_save[:, 0])
#     average_transmissions = np.mean(reward_save[:, 1])
#     average_recovery = np.mean(reward_save[:, 2]) - env.Tf

#     print('Estimated expected reward is {} \n Expected reward is: {}'.format( QTable[env.reset().tobytes()], average_reward))
#     print('Results are: \n Expected number of transmissions: {} \n Expected delay: {}'.format(average_transmissions, average_recovery))
    
#     return(average_reward, average_transmissions, average_recovery)


# results = Test(TransEnv, QTable, 10000)