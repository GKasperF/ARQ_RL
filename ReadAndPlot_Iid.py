import pickle
import numpy as np
from LowerBound.BruteForceUtilityFunctions import lower_convex_hull

store_results = []

#with open('Data/AgentCNNRLresults.pickle', 'rb') as f:
with open('Data/AgentCNNRLresults_Iid_Example_RNN3.pickle', 'rb') as f:
    while 1:
        try:
            store_results = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            break

average_transmissions = [store_results[t][1] for t in range(len(store_results))]
average_recovery = [np.asscalar(store_results[t][2]) for t in range(len(store_results))]

average_recovery = [x for _, x in sorted(zip(average_transmissions, average_recovery))]
average_transmissions.sort()

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




average_transmissions_lb = [convex_hull_results[t][0] for t in range(len(convex_hull_results))]
average_recovery_lb = [convex_hull_results[t][1] for t in range(len(convex_hull_results))]

import matplotlib.pyplot as plt
plt.plot(average_transmissions, average_recovery, 'xk', average_transmissions_heur, average_recovery_heur, '-sg', average_transmissions_lb, average_recovery_lb, '-r')
plt.legend(('Proposed Scheme', 'Multi-burst Transmission', 'Lower Bound'))
plt.xlabel('Average Number of Transmissions')
plt.ylabel('Average Recovery Time')
plt.grid()
plt.show()