import torch
import Envs.PytorchEnvironments as Envs
import copy
import pickle 
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

Num_Samples = 1000000
batch_size = 1

with open('Data/GE_Isolated_Sequence_Example_Tests.pickle', 'rb') as f:
    Channel_Sequence = torch.load(f)

with open('Data/GE_Isolated_Model_Example.pickle', 'rb') as f:
  RNN_Model = torch.load(f)

Tf = 10 

Generator = Envs.GenerateStatesFromErasures(Channel_Sequence, RNN_Model, Tf = torch.tensor(Tf), batch = 1).to(device)

estimate = torch.zeros((Num_Samples, batch_size, Tf))

for i in tqdm(range(Num_Samples)):
  estimate[i, :, :] = Generator.step()

with open('Data/GE_Isolated_StateSequence_Tests.pickle', 'wb') as f:
  torch.save(estimate, f)