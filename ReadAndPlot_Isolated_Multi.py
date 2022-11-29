import pickle
import numpy as np
import torch
from LowerBound.BruteForceUtilityFunctions import lower_convex_hull

store_results = []

#with open('Data/AgentCNNRLresults.pickle', 'rb') as f:
with open('Data/AgentCNNRLResults_MultiPacket_GE_Isolated_Example_NotInOrder.pickle', 'rb') as f:
    while 1:
        try:
            NotInOrderDict = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            break

with open('Data/AgentCNNRLResults_MultiPacket_GE_Isolated_Example.pickle', 'rb') as f:
    while 1:
        try:
            InOrderDict = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            break

pass 

with open('Data/HeuristicsResults_GE_Isolated_Example_InOrder.pickle', 'rb') as f:
    store_results_heur = pickle.load(f)

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

average_transmissions_heur_inorder = [store_results_heur[t][1] for t in range(len(store_results_heur))]
average_recovery_heur_inorder = [store_results_heur[t][2] for t in range(len(store_results_heur))]

test = zip(average_transmissions_heur_inorder, average_recovery_heur_inorder)
test = list(test)
test = lower_convex_hull(test)

average_transmissions_heur_inorder = [test[t][0] for t in range(len(test))]
average_recovery_heur_inorder = [test[t][1] for t in range(len(test))]

average_recovery_heur_inorder = [x for _, x in sorted(zip(average_transmissions_heur_inorder, average_recovery_heur_inorder))]
average_transmissions_heur_inorder.sort()

with open('Data/HeuristicsResults_GE_Isolated_Example.pickle', 'rb') as f:
    store_results_heur = pickle.load(f)

average_transmissions_heur = [store_results_heur[t][1] for t in range(len(store_results_heur))]
average_recovery_heur = [store_results_heur[t][2] for t in range(len(store_results_heur))]

test = zip(average_transmissions_heur, average_recovery_heur)
test = list(test)
test = lower_convex_hull(test)

average_transmissions_heur = [test[t][0] for t in range(len(test))]
average_recovery_heur = [test[t][1] for t in range(len(test))]

average_recovery_heur = [x for _, x in sorted(zip(average_transmissions_heur, average_recovery_heur))]
average_transmissions_heur.sort()

with open('Data/RatelessResults_GE_Isolated_Example.pickle', 'rb') as f:
    store_results_rateless = pickle.load(f)

average_transmissions_rateless = [store_results_rateless[t][0] for t in range(len(store_results_rateless))]
average_recovery_rateless = [store_results_rateless[t][1] for t in range(len(store_results_rateless))]


import matplotlib.pyplot as plt
plt.plot(Transmissions, InOrderDelay, '+k', average_transmissions_heur_inorder, average_recovery_heur_inorder, '--g')
plt.legend(('Proposed Scheme', 'Multi-Burst Transmission'))
plt.xlabel('Average Number of Transmissions')
plt.ylabel('Average In-Order Delay')
plt.grid()
plt.show()

# plt.plot(Transmissions, Delay, 'xk', Transmissions, InOrderDelay, '+k', average_transmissions_heur, average_recovery_heur, '-g', average_transmissions_heur_inorder, average_recovery_heur_inorder, average_transmissions_rateless, average_recovery_rateless, '-b')
# plt.legend(('Delay', 'In-Order Delay', 'Heuristic Recovery Delay', 'Heuristic In-Order Delay', 'Rateless Code Delay'))
# plt.xlabel('Average Number of Transmissions')
# plt.ylabel('Average Delay')
# plt.ylim((0, 25))
# plt.xlim(right=6)
# plt.grid()
# plt.show()