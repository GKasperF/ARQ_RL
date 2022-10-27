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

  time_save = torch.empty((0, 1)).to(device)
  for i in range(Nit):
      done = 0
      reward_acc[:] = 0
      transmissions[:] = 0
      time_instant[:] = 1
      number_successes[:] = 0
      SuccessF = torch.zeros(batch).to(device)
      if i == 0:
        state = env.reset()
      else:
        state = env.reset()
        estimate = env.ChannelModel_Sequence[i-1, :, :]
        state[SuccessF == 0, env.Tf:] = estimate[SuccessF == 0, :]
      env.index = i
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
      time_save = torch.cat((time_save, copy.deepcopy(time_instant.reshape(batch, 1))))

  TimeReceived = torch.arange(start=0, end=Nit)
  time_save = time_save.reshape(TimeReceived.shape)

  TimeReceived = TimeReceived + time_save
  for i in range(Nit-1):
    TimeReceived[i+1] = torch.maximum(TimeReceived[i], TimeReceived[i+1])

  AvgInOrder = torch.mean(TimeReceived) - env.Tf



  # average_reward = torch.mean(reward_save[:, 0])
  # average_transmissions = torch.mean(reward_save[:, 1])
  # average_recovery = torch.mean(reward_save[:, 2]) - env.Tf

  #print('Estimated expected reward is {} \n Expected reward is: {}'.format(Q(env.reset()), average_reward))
  
  return(AvgInOrder)



with open('Data/Iid_StateSequence_Tests.pickle', 'rb') as f:
  ChannelModel_Sequence = torch.load(f)
with open('Data/Iid_Sequence_Example_Tests.pickle', 'rb') as f:
  Channel_Erasures = torch.load(f)


batch = 1
#Channel = Envs.iidchannel(0.1, batch).to(device)
#Channel = Envs.GilbertElliott(0.25, 0.25, 0, 1, batch).to(device)
TransEnv = Envs.EnvFeedbackRNN_ReadFromFile_MultiPackets(10, 0.6, 5, Channel_Erasures, ChannelModel_Sequence, batch)
TransEnv = TransEnv.to(device)


class CPU_Unpickler(pickle.Unpickler):
  def find_class(self, module, name):
      if module == 'torch.storage' and name == '_load_from_bytes':
          return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
      else: return super().find_class(module, name)


all_results_dict = {}
all_results_list = []

path = 'Data/ModelCNN_Iid_Example_RNN*.pickle'
for filename in glob.glob(path):
  with open(filename, 'rb') as f:
    Q = CPU_Unpickler(f).load()
  Q = Q.to(device)
  t0 = time()
  print(filename)
  AvgInOrder = Test(TransEnv, Q, 100000, batch)
  t1 = time()

  print('Testing takes {} seconds'.format(t1-t0))
  all_results_dict[filename] = AvgInOrder
  all_results_list.append(AvgInOrder)
  try:
    with open('Data/AgentCNNRLResults_MultiPacket_Iid_Example.pickle', 'wb') as f:
      pickle.dump(all_results_dict, f)
  except Exception as e:
    with open('Data/AgentCNNRLResults_MultiPacket_Iid_Example.pickle', 'wb') as f:
      pickle.dump(all_results_list, f)

try:
  with open('Data/AgentCNNRLResults_MultiPacket_Iid_Example.pickle', 'wb') as f:
    pickle.dump(all_results_dict, f)
except Exception as e:
  with open('Data/AgentCNNRLResults_MultiPacket_Iid_Example.pickle', 'wb') as f:
    pickle.dump(all_results_list, f)