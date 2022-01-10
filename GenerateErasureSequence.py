import torch
import Envs.PytorchEnvironments as Envs
import copy
import pickle 
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# class Fritchman():
#     def __init__(self, alpha, beta, epsilon, M, batch = 1):

Num_Samples = 10000000
batch_size = 1

Channel_Sequence = torch.zeros(Num_Samples).to(device)

Channel = Envs.GilbertElliott(0.25, 0.25, 0, 1, batch_size).to(device)
#Channel = Envs.Fritchman(0.25, 1, 0, 5, batch_size).to(device)
for i in range(Num_Samples):
  Channel_Sequence[i] = Channel.step()

with open('Data/Fritchman_Sequence.pickle', 'wb') as f:
    torch.save(Channel_Sequence, f)