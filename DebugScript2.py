import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('Data/Fritchman_Sequence_Example_ForTests.pickle', 'rb') as f:
    Channel_Sequence = torch.load(f)

with open('Data/Fritchman_Sequence_States_Example_ForTests.pickle', 'rb') as f:
    Current_State = torch.load(f)

Tf = 10
total_transmissions = len(Current_State[:, Tf])

for i in range(Tf, Tf + 20):
  prob_successes = sum(Channel_Sequence[:, i] == 1)/total_transmissions
  print('Prob for i = {} is {}'.format(i, prob_successes))
  temp_erasures = Channel_Sequence[:, i] == 0
  Channel_Sequence = Channel_Sequence[temp_erasures, :]


for i in range(Tf, Tf+6):
  print('Current time instant is {}'.format(i - Tf))
  p0 = sum(Current_State[:, i] == 0)/len(Current_State[:, i])
  p1 = sum(Current_State[:, i] == 1)/len(Current_State[:, i])
  p2 = sum(Current_State[:, i] == 2)/len(Current_State[:, i])
  p3 = sum(Current_State[:, i] == 3)/len(Current_State[:, i])

  print('Probability of current state is:{}\n'.format((p0, p1, p2, p3)))

  prob_successes = sum(Channel_Sequence[:, i] == 1)/len(Current_State[:, i])
  print('Prob of success is: {}\n'.format(prob_successes))

  temp_erasures = Channel_Sequence[:, i] == 0
  Current_State = Current_State[temp_erasures, :]
  Channel_Sequence = Channel_Sequence[temp_erasures, :]

  num_erasures = len(Channel_Sequence[:, i])

  temp0 = Current_State[:, i] == 0
  temp1 = Current_State[:, i] == 1
  temp2 = Current_State[:, i] == 2
  temp3 = Current_State[:, i] == 3

  p0 = sum(temp0)/num_erasures
  p1 = sum(temp1)/num_erasures
  p2 = sum(temp2)/num_erasures
  p3 = sum(temp3)/num_erasures

  print('Probability of current state given erasure is: {}\n'.format((p0, p1, p2, p3)))

pass