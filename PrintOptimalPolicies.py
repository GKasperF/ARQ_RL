import pickle
import copy
from LowerBound.BruteForceUtilityFunctions import convex_hull, lower_convex_hull
from LowerBound.BinaryConversion import dectobin
import numpy as np

with open('Data/BruteForce_GE_Isolated_Example.pickle', 'rb') as f:
    store_results_brute_force = pickle.load(f)


AllPolicies = copy.deepcopy(store_results_brute_force)

convex_hull_results = lower_convex_hull(store_results_brute_force)
i = 0
while 1:
	point = convex_hull_results[i]
	#if point[0] < (1/(1 - 0.5) - 0.05):
	if point[0] < (1 - 0.05):
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

policies = range(1, 2**21)

optimal_indices = []
optimal_policies = []

for point in convex_hull_results:
	index_temp = AllPolicies.index(point)
	optimal_indices.append(index_temp)
	policy_temp = policies[AllPolicies.index(point)]
	policy_vec_temp = dectobin(policy_temp, 21)
	policy_vec_temp = policy_vec_temp.reshape(21)
	optimal_policies.append(policy_vec_temp)
	print(policy_vec_temp)


convex_hull_results = np.array(convex_hull_results)
reward_multiplier = np.array([[-1.0], [-2.0]])

rewards = np.matmul(convex_hull_results, reward_multiplier)
best = np.argmax(rewards)
print(convex_hull_results[best])
print(optimal_policies[best])