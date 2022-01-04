import torch
import Envs.PytorchEnvironments as Envs
import copy
import pickle 
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ChannelModel(torch.nn.Module):
  def __init__(self, hidden_size, num_layers, output_size):
    super(ChannelModel, self).__init__()

    self.Layer1 = torch.nn.LSTM(input_size = 2, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
    self.FinalLayer = torch.nn.Linear(hidden_size, output_size)
    self.prob_layer = torch.nn.Sigmoid()

  def forward(self, x, h, c):
    L1, (h_out, c_out) = self.Layer1(x, (h, c))
    L2 = self.FinalLayer(L1)
    output = self.prob_layer(L2)
    #output = torch.sigmoid(L2)

    return output, (h_out, c_out)

# with open('Data/SaveLossRNN.pickle', 'rb') as f:
#   save_loss = torch.load(f)
#save_loss = save_loss[:-1]

with open('Data/RNN_Model_Test3.pickle', 'rb') as f:
  RNN_Model = torch.load(f)

hidden_size = 10
num_layers = 5
batch_size = 1
Tf = 10

h_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
c_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)

state_in0 = torch.tensor([0.0, 0.0]).to(device).reshape((batch_size, 1, 2))
state_in1 = torch.tensor([1.0, 1.0]).to(device).reshape((batch_size, 1, 2))
state_in2 = torch.tensor([0.0, 1.0]).to(device).reshape((batch_size, 1, 2))


estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in2, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)


import matplotlib.pyplot as plt
plt.plot(range(len(save_loss)), save_loss)
plt.show()

pass