import torch
import Envs.PytorchEnvironments as Envs
import copy
import pickle 
import numpy as np

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
Num_Samples = 10000000
UpdateSteps = 10000

with open('Data/Channel_Sequence.pickle', 'rb') as f:
  #Channel_Sequence = pickle.load(f).to(device)
  Channel_Sequence = torch.load(f).to(device)

h_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
c_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
optimizer.zero_grad()

#save_loss = torch.zeros(int((Num_Samples - Tf)/UpdateSteps)+1).to(device)
save_loss = np.zeros(int((Num_Samples - Tf)/UpdateSteps)+1)
j=0
temp = 0.5
prob_trans = temp*torch.ones(1).to(device)
for i in range(Num_Samples - Tf):
  target = Channel_Sequence[i + 1 : i + 1 + Tf]
  transmission = torch.bernoulli(prob_trans).type(torch.uint8)
  state_in = torch.cat( ((Channel_Sequence[i].type(torch.uint8) & transmission).type(torch.float).reshape((batch_size, 1, 1)), transmission.type(torch.float).reshape((batch_size, 1, 1))), dim=2)
  estimate, (h_out, c_out) = RNN_Model(state_in, h_in, c_in)
  loss = criterion(estimate, target.detach().reshape(estimate.shape))
  loss.backward()
  h_in = h_out.detach()
  c_in = c_out.detach()
  if (i % UpdateSteps == 0):
    optimizer.step()
    optimizer.zero_grad()
    save_loss[j] = loss.detach().to('cpu').numpy()
    j+=1
    print(i, j)

with open('Data/SaveLossRNN_GE.pickle', 'wb') as f:
  torch.save(save_loss, f)

with open('Data/RNN_Model_GE.pickle', 'wb') as f:
  torch.save(RNN_Model, f)

# for i in range(Num_Attempts):
#     optimizer.zero_grad()
#     State_in = Channel.step().type(torch.float).reshape((batch_size, 1, 1))

#     h_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
#     c_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
#     for j in range(Num_Attempts2):
#         State_out, (h_out, c_out) = RNN_Model(State_in, h_in, c_in)
#         Outcome = Channel.step().type(torch.float)
#         loss = criterion(State_out, Outcome.reshape( State_out.shape).detach())
#         #optimizer.zero_grad()
#         loss.backward()
#         h_in = h_out.detach()
#         c_in = c_out.detach()
#         #State_in = Outcome.detach().reshape(State_in.shape)
#         State_in = State_out.detach()

#     optimizer.step()

# Channel_state = torch.bernoulli(torch.tensor( 0.5*torch.ones((batch_size, 2)))).to(device)
# Channel_state[:, 1] = 1 - Channel_state[:, 0]

# State_in = Channel_state[:,0].reshape((batch_size, 1, 1))

# h_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
# c_in = torch.zeros((num_layers, batch_size, hidden_size)).to(device)
# for j in range(Num_Attempts2):
#     State_out, (h_out, c_out) = RNN_Model(State_in, h_in, c_in)
#     Channel_state = torch.matmul(Channel_state, Transition_Matrix)
#     Actual_Prob = Channel_state[:, 0]
#     print('Prediction is: {} \n Actual is: {}'.format(State_out, Actual_Prob))
#     h_in = h_out.detach()
#     c_in = c_out.detach()
#     State_in = State_out.detach()