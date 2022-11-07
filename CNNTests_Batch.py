import torch
import Envs.PytorchEnvironments as Envs
import copy
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

CUDA_LAUNCH_BLOCKING=1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ChannelModel(torch.nn.Module):
  def __init__(self, num_inputs, hidden_size, output_size):
    super(ChannelModel, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_inputs = num_inputs
    self.output_size = output_size

    self.Layer1 = torch.nn.Linear(num_inputs*2, hidden_size*num_inputs*2)
    self.Layer2 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size = 3, padding=1)
    self.Layer2 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size = 3, padding=1)
    self.Layer3 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size = 3, padding=1)
    self.Layer4 = torch.nn.Conv1d(hidden_size, 1, kernel_size = 1, padding=0)
    #self.Layer1 = torch.nn.LSTM(input_size = 2, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
    self.FinalLayer = torch.nn.Linear(num_inputs*2, output_size)
    self.prob_layer = torch.nn.Sigmoid()

  def forward(self, x):
    x = x.reshape(-1, self.num_inputs * 2)
    L1 = self.Layer1(x)
    L1_conv = L1.view(-1, self.hidden_size, self.num_inputs*2)
    ReLU1 = F.relu(L1_conv)
    L2 = self.Layer2(ReLU1)
    ReLU2 = F.relu(L2)
    L3 = self.Layer3(ReLU2)
    ReLU3 = F.relu(L3)
    L4 = self.Layer4(ReLU3)
    ReLU4 = F.relu(L4)
    ReLU4 = ReLU4.view(-1, self.num_inputs*2)
    L5 = self.FinalLayer(ReLU4)
    output = self.prob_layer(L5)

    return output

hidden_size = 10
num_layers = 5
batch_size = 1000
Tf = 10

num_inputs = 20

RNN_Model = ChannelModel(num_inputs = num_inputs, hidden_size = hidden_size, output_size = Tf).to(device)

#criterion = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.BCELoss(reduction='mean')
#Params_LSTM = list(RNN_Model.Layer1.parameters())
#Params_Linear = list(RNN_Model.FinalLayer.parameters())
optimizer = torch.optim.Adam(list(RNN_Model.parameters()))
UpdateSteps = 1

with open('Data/Fritchman_Sequence_Example.pickle', 'rb') as f:
  Channel_Sequence_All = torch.load(f).to(device)

Num_Samples = int( Channel_Sequence_All.shape[0]*Channel_Sequence_All.shape[1] / batch_size)
Channel_Sequence_All = Channel_Sequence_All.reshape(batch_size, Num_Samples).to(device)

update_count = 0
optimizer.zero_grad()
j=0
save_loss = np.zeros(int(((Num_Samples - Tf - num_inputs))/1)+1)

temp = 0.5
prob_trans = temp*torch.ones((batch_size, num_inputs)).to(device)
for i in range(Num_Samples - Tf - num_inputs):
    target = Channel_Sequence_All[:, i + num_inputs : i + num_inputs + Tf]
    transmission = torch.bernoulli(prob_trans).type(torch.uint8)
    state_in = torch.cat( ((Channel_Sequence_All[:, i:i+num_inputs].type(torch.uint8) & transmission).type(torch.float).reshape((batch_size, num_inputs, 1)), transmission.type(torch.float).reshape((batch_size, num_inputs, 1))), dim=2)
    #estimate, (h_out, c_out) = RNN_Model(state_in, h_in, c_in)
    estimate = RNN_Model(state_in)
    loss = criterion(estimate, target.reshape(estimate.shape).detach())

    try:
      save_loss[i] = loss
      #loss_copy = copy.copy(loss).detach().to('cpu').numpy()
      #save_loss[i] = loss_copy
    except Exception as e:
      print(e)
      pass
    optimizer.zero_grad()
    try:
      loss.backward()
    except Exception as e:
      print(e)
      pass
      
    optimizer.step()

    if (update_count % UpdateSteps == 0):
        j+=1
        print(update_count, j)
    update_count = update_count + 1

#save_loss = save_loss.detach().cpu().numpy()

with open('Data/Fritchman_Loss_CNN_Example.pickle', 'wb') as f:
  torch.save(save_loss, f)

#with open('Data/RNN_Model_GE_Isolated_Erasures_Batch.pickle', 'wb') as f:
with open('Data/Fritchman_Model_CNN_Example.pickle', 'wb') as f:
  torch.save(RNN_Model, f)


save_loss = save_loss[:-1]
plt.plot(range(len(save_loss)), save_loss)
plt.xlabel('Step')
plt.ylabel('BCE Loss')
plt.show()

# RNN_Model.Layer1.proj_size = 0
# RNN_Model = RNN_Model.to(device)

# batch_size = 1

# state_in0 = torch.tensor([0.0, 0.0]).to(device).reshape((batch_size, 1, 2))
# state_in1 = torch.tensor([1.0, 1.0]).to(device).reshape((batch_size, 1, 2))
# state_in2 = torch.tensor([0.0, 1.0]).to(device).reshape((batch_size, 1, 2))

# state_in = torch.cat((state_in1, state_in1, state_in1, state_in1, state_in1), dim=1)

# estimate = RNN_Model(state_in)
# print(estimate)

# state_in = torch.cat((state_in2, state_in0, state_in0, state_in0, state_in0), dim=1)

# estimate = RNN_Model(state_in)
# print(estimate)

# state_in = torch.cat((state_in0, state_in0, state_in0, state_in0, state_in2), dim=1)

# estimate = RNN_Model(state_in)
# print(estimate)

# state_in = torch.cat((state_in0, state_in0, state_in0, state_in1, state_in2), dim=1)

# estimate = RNN_Model(state_in)
# print(estimate)

# state_in = torch.cat((state_in0, state_in0, state_in0, state_in2, state_in1), dim=1)

# estimate = RNN_Model(state_in)
# print(estimate)

# state_in = torch.cat((state_in0, state_in0, state_in0, state_in0, state_in0), dim=1)

# estimate = RNN_Model(state_in)
# print(estimate)

# state_in = torch.cat((state_in1, state_in0, state_in0, state_in0, state_in0), dim=1)

# estimate = RNN_Model(state_in)
# print(estimate)

# state_in = torch.cat((state_in1, state_in1, state_in1, state_in1, state_in2), dim=1)

# estimate = RNN_Model(state_in)
# print(estimate)