import numpy as np
from time import time
from joblib import Parallel, delayed
import multiprocessing
import pickle
import Envs.PytorchEnvironments as EnvsNN
import torch
import Envs.Environments as Envs
import dill
import io

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


num_cores = 2
ChannelNNMC = EnvsNN.GilbertElliott(1, 0, 0, 1, 1).to(device)
TransEnvNNMC = EnvsNN.EnvFeedbackGeneral(10, 1.4, 5, ChannelNNMC, 1)
TransEnvNNMC = TransEnvNNMC.to(device)

ChannelNN = EnvsNN.GilbertElliott(1, 0, 0, 1, 1).to(device)
TransEnvNN = EnvsNN.EnvFeedbackGeneral(10, 1.4, 5, ChannelNN, 1)
TransEnvNN = TransEnvNN.to(device)

Channel = Envs.GilbertElliott(1, 0, 0, 1)
TransEnv = Envs.EnvFeedbackGeneral(10, 1.4, 5, Channel)

with open('Data/SaveModelNNMC.pickle', 'rb') as f:
  QNNMC = pickle.load(f)

# with open('Data/SaveModelAlpha14.pickle', 'rb') as f:
#   QNN = pickle.load(f)

class CPU_Unpickler(pickle.Unpickler):
  def find_class(self, module, name):
      if module == 'torch.storage' and name == '_load_from_bytes':
          return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
      else: return super().find_class(module, name)


with open('Data/ModelCNNBatch2.3.pickle', 'rb') as f:
  QNN = CPU_Unpickler(f).load()

with open('Data/SaveModelTable.pickle', 'rb') as f:
  QTable = dill.load(f)

QNNMC = QNNMC.to(device)
stateNNMC = TransEnvNNMC.reset()

QNN = QNN.to(device)
stateNN = TransEnvNN.reset()

stateTable = TransEnv.reset()

SNN = torch.zeros((30))
STable = np.zeros((30))
SNNMC = torch.zeros((30))



for t in range(30):
  action_indexTable = np.argmax(QTable[stateTable.tobytes()])
  STable[t] = action_indexTable
  stateTable, _, _, _ = TransEnv.step(TransEnv.actions[action_indexTable])

  action_indexNN = torch.argmax(QNN(stateNN), dim = 1)
  SNN[t] = action_indexNN
  stateNN, reward, done, SuccessF = TransEnvNN.step(action_indexNN.reshape(1,1))

  action_indexNNMC = torch.argmax(QNNMC(stateNNMC), dim = 1)
  SNNMC[t] = action_indexNNMC
  stateNNMC, reward, done, SuccessF = TransEnvNNMC.step(TransEnvNNMC.actions[action_indexNNMC].reshape(1,1))
#breakpoint()

print('Policy learned by NNMC: {}'.format(SNNMC))
print('Policy learned by NN: {}'.format(SNN))
print('Policy learned by Table: {}'.format(STable))


stateNNMC = TransEnvNNMC.reset()
stateNN = TransEnvNN.reset()
stateTable = TransEnv.reset()

for t in range(30):
  print('Current state: {}'.format(stateTable))
  print('Q value for NNMC: {}'.format( QNNMC(stateNNMC)))
  print('Q value for NN: {}'.format( QNN(stateNN)))
  print('Q value for Table: {}'.format( QTable[stateTable.tobytes()]))
  print('\n \n')

  action_indexTable = np.argmax(QTable[stateTable.tobytes()])
  stateTable, _, _, _ = TransEnv.step(TransEnv.actions[action_indexTable])
  stateNN, reward, done, SuccessF = TransEnvNN.step(TransEnvNN.actions[action_indexTable].reshape(1,1))
  stateNNMC, reward, done, SuccessF = TransEnvNNMC.step(TransEnvNNMC.actions[action_indexTable].reshape(1,1))

print('done')