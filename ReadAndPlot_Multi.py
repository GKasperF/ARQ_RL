import pickle
import numpy as np
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

# import matplotlib.pyplot as plt
# plt.plot(average_transmissions, average_recovery, 'xk', average_transmissions2, average_recovery2, 'xb', average_transmissions_heur, average_recovery_heur, '-sg', average_transmissions_lb, average_recovery_lb, '-or')
# plt.legend(('Proposed Scheme', 'Q-learning with Lookup Table', 'Multi-burst Transmission', 'Lower Bound'))
# plt.xlabel('Average Number of Transmissions')
# plt.ylabel('Average Recovery Time')
# plt.grid()
# plt.show()