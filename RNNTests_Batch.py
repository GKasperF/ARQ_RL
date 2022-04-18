import torch
import Envs.PytorchEnvironments as Envs
import copy
import pickle 
import numpy as np
import matplotlib.pyplot as plt

from GenerateErasureSequence import Channel

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
batch_size = 1000
Tf = 10

RNN_Model = ChannelModel(hidden_size = hidden_size, num_layers = num_layers, output_size = Tf).to(device)
#criterion = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.BCELoss(reduction='mean')
Params_LSTM = list(RNN_Model.Layer1.parameters())
Params_Linear = list(RNN_Model.FinalLayer.parameters())
optimizer = torch.optim.Adam(Params_LSTM + Params_Linear)
UpdateSteps = 1

with open('Data/Iid_Sequence_Example.pickle', 'rb') as f:
  Channel_Sequence_All = torch.load(f).to(device)
  # Channel_Sequence_All = Channel_Sequence_All[:, 0:1000]
  # Channel_Sequence_All = Channel_Sequence_All.repeat(1, 1)
# with open('Data/TraceSets/TraceUFSC_Failures.pth', 'rb') as f:
#   Channel_Sequence_All = torch.load(f).to(device)
#   Channel_Sequence_All = Channel_Sequence_All.repeat(610)
#   Channel_Sequence_All = Channel_Sequence_All[0:10000000].reshape(10000, 1000)

#Num_Samples = Channel_Sequence_All.shape[1]
Num_Samples = int( Channel_Sequence_All.shape[0]*Channel_Sequence_All.shape[1] / batch_size)
Channel_Sequence_All = Channel_Sequence_All.reshape(batch_size, Num_Samples).to(device)

update_count = 0
optimizer.zero_grad()
j=0
save_loss = np.zeros(int(((Num_Samples - Tf))/1)+1)

h_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
c_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)

temp = 0.5
prob_trans = temp*torch.ones(batch_size).to(device)
for i in range(Num_Samples - Tf):
    target = Channel_Sequence_All[:, i + 1 : i + 1 + Tf]
    transmission = torch.bernoulli(prob_trans).type(torch.uint8)
    state_in = torch.cat( ((Channel_Sequence_All[:, i].type(torch.uint8) & transmission).type(torch.float).reshape((batch_size, 1, 1)), transmission.type(torch.float).reshape((batch_size, 1, 1))), dim=2)
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

with open('Data/Iid_Loss_Example.pickle', 'wb') as f:
  torch.save(save_loss, f)

#with open('Data/RNN_Model_GE_Isolated_Erasures_Batch.pickle', 'wb') as f:
with open('Data/Iid_Model_Example.pickle', 'wb') as f:
  torch.save(RNN_Model, f)


save_loss = save_loss[:-1]
plt.plot(range(len(save_loss)), save_loss)
plt.xlabel('Step')
plt.ylabel('BCE Loss')
plt.show()

RNN_Model.Layer1.proj_size = 0
RNN_Model = RNN_Model.to(device)

batch_size = 1

h_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
c_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)


state_in0 = torch.tensor([0.0, 0.0]).to(device).reshape((batch_size, 1, 2))
state_in1 = torch.tensor([1.0, 1.0]).to(device).reshape((batch_size, 1, 2))
state_in2 = torch.tensor([0.0, 1.0]).to(device).reshape((batch_size, 1, 2))

estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in1, h_in, c_in)
print(estimate)
print('Starting erasure burst')
estimate, (h_in, c_in) = RNN_Model(state_in2, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in2, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in2, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in2, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in2, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in2, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)
estimate, (h_in, c_in) = RNN_Model(state_in0, h_in, c_in)
print(estimate)