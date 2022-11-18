import pickle
import numpy as np
from LowerBound.BruteForceUtilityFunctions import lower_convex_hull
import torch 
import io

class CPU_Unpickler(pickle.Unpickler):
  def find_class(self, module, name):
      if module == 'torch.storage' and name == '_load_from_bytes':
          return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
      else: return super().find_class(module, name)

with open('Data/AgentCNNRLresults_Iid_Example_RNN_Dict.pickle', 'rb') as f:
    result_dict = CPU_Unpickler(f).load()

#return(average_reward, average_transmissions, average_recovery)

average_transmissions = [result_dict[model][1] for model in result_dict]
average_recovery = [result_dict[model][2] for model in result_dict]

average_recovery = [x for _, x in sorted(zip(average_transmissions, average_recovery))]
average_transmissions.sort()

with open('Data/AgentCNN_LSTM_RLresultsTestBatch_Iid.pickle', 'rb') as f:
    result_dict_DRQN = CPU_Unpickler(f).load()

average_transmissions_DRQN = [result_dict_DRQN[model][1] for model in result_dict_DRQN]
average_recovery_DRQN = [result_dict_DRQN[model][2] for model in result_dict_DRQN]

# test = zip(average_transmissions2, average_recovery2)
# test = list(test)
# test = lower_convex_hull(test)
# i = 1
# while 1:
# 	if i == len(test):
# 		break
# 	point = test[i]
# 	previous_point = test[i-1]
# 	if point[0] > previous_point[0] and point[1] > previous_point[1]:
# 		test.pop(i)
# 	else:
# 		i+=1

# average_transmissions2 = [test[t][0] for t in range(len(test))]
# average_recovery2 = [test[t][1] for t in range(len(test))]

average_recovery_DRQN = [x for _, x in sorted(zip(average_transmissions_DRQN, average_recovery_DRQN))]
average_transmissions_DRQN.sort()

with open('Data/HeuristicsResults_iid_Example.pickle', 'rb') as f:
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

with open('Data/BruteForceIid_Example.pickle', 'rb') as f:
    store_results_brute_force = pickle.load(f)

convex_hull_results = lower_convex_hull(store_results_brute_force)
i = 0
while 1:
    point = convex_hull_results[i]
    if point[0] < (1/(1 - 0.5) - 0.05):
        convex_hull_results.pop(i)
    else:
        i+=1
    
    if i == len(convex_hull_results):
        break


average_transmissions_lb = [convex_hull_results[t][0] for t in range(len(convex_hull_results))]
average_recovery_lb = [convex_hull_results[t][1] for t in range(len(convex_hull_results))]

import matplotlib.pyplot as plt
plt.plot(average_transmissions, average_recovery, 'xk', average_transmissions_DRQN, average_recovery_DRQN, 'xb', average_transmissions_heur, average_recovery_heur, '-sg', average_transmissions_lb, average_recovery_lb, '-or')
plt.legend(('Proposed Scheme', 'DRQN', 'Multi-burst Transmission', 'Lower Bound'))
plt.xlabel('Average Number of Transmissions')
plt.ylabel('Average Recovery Time')
plt.grid()
plt.show()