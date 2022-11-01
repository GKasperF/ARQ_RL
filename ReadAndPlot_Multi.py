import pickle
import numpy as np
import torch
from LowerBound.BruteForceUtilityFunctions import lower_convex_hull

store_results = []

#with open('Data/AgentCNNRLresults.pickle', 'rb') as f:
with open('Data/AgentCNNRLResults_MultiPacket_Fritchman_Example_NotInOrder.pickle', 'rb') as f:
    while 1:
        try:
            NotInOrderDict = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            break

with open('Data/AgentCNNRLResults_MultiPacket_Fritchman_Example.pickle', 'rb') as f:
    while 1:
        try:
            InOrderDict = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            break

pass 

temp = torch.arange(start=0, end=100000).type(torch.float)
avg_cancel = torch.mean(temp)

InOrderDelay = torch.zeros(len(InOrderDict))
Delay = torch.zeros(len(NotInOrderDict))
Transmissions = torch.zeros(len(NotInOrderDict))

i = 0
for key in NotInOrderDict:
    InOrderDelay[i] = InOrderDict[key].cpu() - avg_cancel 
    Delay[i] = NotInOrderDict[key][0].cpu() - avg_cancel
    Transmissions[i] = NotInOrderDict[key][1].cpu()
    i = i + 1


import matplotlib.pyplot as plt
plt.plot(Transmissions, Delay, 'xk', Transmissions, InOrderDelay, 'xb')
plt.legend(('Delay', 'In-order Delay'))
plt.xlabel('Average Number of Transmissions')
plt.ylabel('Average Delay')
plt.grid()
plt.show()