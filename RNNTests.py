import torch
import Envs.PytorchEnvironments as Envs
import copy
import pickle 
import numpy as np
import matplotlib.pyplot as plt

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

hidden_size = 10
num_layers = 5
batch_size = 1
Tf = 10

RNN_Model = ChannelModel(hidden_size = hidden_size, num_layers = num_layers, output_size = Tf).to(device)
criterion = torch.nn.MSELoss(reduction='mean')
Params_LSTM = list(RNN_Model.Layer1.parameters())
Params_Linear = list(RNN_Model.FinalLayer.parameters())
optimizer = torch.optim.Adam(Params_LSTM + Params_Linear)
UpdateSteps = 1000

#with open('Data/TraceSets/distance_10m/Run3_10m.torch', 'rb') as f:
with open('Data/GE_Sequence_Isolated.pickle', 'rb') as f:
  Channel_Sequence_All = torch.load(f).to(device)

#Channel_Sequence_All = Channel_Sequence_All[0:1000000]

# with open('Data/TraceSets/erasure80.bin', 'rb') as f:
#   test = f.read()  

# Channel_Sequence_All = torch.zeros(len(test))

# for i in range(len(test)):
#   Channel_Sequence_All[i] = 1 - test[i]

# Channel_Sequence_All = Channel_Sequence_All.repeat(30)
# Channel_Sequence_All = Channel_Sequence_All[0:10000000]

# rand_perm = torch.randperm(len(Channel_Sequence_All))

# Channel_Sequence_All = Channel_Sequence_All[rand_perm, :]

Num_Samples = len(Channel_Sequence_All)
Channel_Sequence_All = Channel_Sequence_All.reshape(1, len(Channel_Sequence_All)).to(device)

update_count = 0
optimizer.zero_grad()
j=0
#save_loss = np.zeros(int(((Num_Samples - Tf)*len(Channel_Sequence_All))/UpdateSteps)+1)
save_loss = np.zeros(int(((Num_Samples - Tf)*len(Channel_Sequence_All))/1)+1)


for k in range(len(Channel_Sequence_All)):
  Channel_Sequence = Channel_Sequence_All[k]

  h_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
  c_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
  
  temp = 1.0
  prob_trans = temp*torch.ones(1).to(device)
  for i in range(Num_Samples - Tf):
    target = Channel_Sequence[i + 1 : i + 1 + Tf]
    transmission = torch.bernoulli(prob_trans).type(torch.uint8)
    state_in = torch.cat( ((Channel_Sequence[i].type(torch.uint8) & transmission).type(torch.float).reshape((batch_size, 1, 1)), transmission.type(torch.float).reshape((batch_size, 1, 1))), dim=2)
    estimate, (h_out, c_out) = RNN_Model(state_in, h_in, c_in)
    loss = criterion(estimate, target.detach().reshape(estimate.shape))
    loss.backward()
    save_loss[i] = loss.detach().to('cpu').numpy()
    h_in = h_out.detach()
    c_in = c_out.detach()
    if (update_count % UpdateSteps == 0):
      optimizer.step()
      optimizer.zero_grad()
      j+=1
      print(update_count, j)
    update_count = update_count + 1

with open('Data/SaveLossRNN_GE_Isolated_AllSeeing.pickle', 'wb') as f:
  torch.save(save_loss, f)

with open('Data/RNN_Model_GE_Isolated_AllSeeing.pickle', 'wb') as f:
  torch.save(RNN_Model, f)