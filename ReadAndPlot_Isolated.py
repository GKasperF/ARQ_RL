import pickle
import numpy as np
from LowerBound.BruteForceUtilityFunctions import lower_convex_hull

store_results = []

#with open('Data/AgentCNNRLresults.pickle', 'rb') as f:
with open('Data/AgentCNNRLresultsTestBatch_GE_Isolated_RNN.pickle', 'rb') as f:
    while 1:
        try:
            store_results = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            break

average_transmissions = [store_results[t][1] for t in range(len(store_results))]
average_recovery = [np.asscalar(store_results[t][2]) for t in range(len(store_results))]

average_recovery = [x for _, x in sorted(zip(average_transmissions, average_recovery))]
average_transmissions.sort()

# with open('Data/AgentRLresults_Memory_FewEpisodes.pickle', 'rb') as f:
#     store_results2 = pickle.load(f)

# average_transmissions2 = [store_results2[t][1] for t in range(len(store_results2))]
# average_recovery2 = [store_results2[t][2] for t in range(len(store_results2))]

# average_recovery2 = [x for _, x in sorted(zip(average_transmissions2, average_recovery2))]
# average_transmissions2.sort()

with open('Data/HeuristicsResults_GE_Isolated.pickle', 'rb') as f:
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

# with open('Data/BruteForceGESimple.pickle', 'rb') as f:
#     store_results_brute_force = pickle.load(f)

# convex_hull_results = lower_convex_hull(store_results_brute_force)
# i = 0
# while 1:
#     point = convex_hull_results[i]
#     if point[0] < (1/(1 - 0.5) - 0.05):
#         convex_hull_results.pop(i)
#     else:
#         i+=1
    
#     if i == len(convex_hull_results):
#         break


# average_transmissions_lb = [convex_hull_results[t][0] for t in range(len(convex_hull_results))]
# average_recovery_lb = [convex_hull_results[t][1] for t in range(len(convex_hull_results))]

with open('Data/BruteForceGEFull_BadStart.pickle', 'rb') as f:
    store_results_brute_force = pickle.load(f)

convex_hull_results = lower_convex_hull(store_results_brute_force)
i = 0
while 1:
    point = convex_hull_results[i]
    if point[0] < (1/(1 - 0.086) - 0.05):
        convex_hull_results.pop(i)
    else:
        i+=1
    
    if i == len(convex_hull_results):
        break

i = 1
while 1:
	if i == len(convex_hull_results):
		break
	point = convex_hull_results[i]
	previous_point = convex_hull_results[i-1]
	if point[0] > previous_point[0] and point[1] > previous_point[1]:
		convex_hull_results.pop(i)
	else:
		i+=1

# for i in range(1, len(convex_hull_results)):
#     point = convex_hull_results[i]
#     previous_point = convex_hull_results[i-1]

#     if point[0] > previous_point[0] and point[1] > previous_point[1]:




average_transmissions_lb = [convex_hull_results[t][0] for t in range(len(convex_hull_results))]
average_recovery_lb = [convex_hull_results[t][1] for t in range(len(convex_hull_results))]

import matplotlib.pyplot as plt
plt.plot(average_transmissions, average_recovery, 'xk', average_transmissions_heur, average_recovery_heur, '-sg', average_transmissions_lb, average_recovery_lb, '-r')
plt.legend(('Neural Network (With RNN)', 'Heuristic', 'Lower Bound'))
plt.xlabel('Average Number of Transmissions')
plt.ylabel('Average Recovery Time')
plt.grid()
plt.show()