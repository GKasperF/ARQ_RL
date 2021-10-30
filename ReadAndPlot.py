import pickle

with open('Data/AgentRLresults.pickle', 'rb') as f:
    store_results = pickle.load(f)


alpha_range = np.arange(0.5, 1.6, 0.1)
average_transmissions = [store_results[t][1] for t in range(len(store_results))]
average_recovery = [store_results[t][2] for t in range(len(store_results))]

average_recovery = [x for _, x in sorted(zip(average_transmissions, average_recovery))]
alpha_range = [x for _, x in sorted(zip(average_transmissions, alpha_range))]
average_transmissions.sort()

with open('Data/HeuristicsResults.pickle', 'rb') as f:
    store_results_heur = pickle.load(f)

average_transmissions_heur = [store_results_heur[t][1] for t in range(len(store_results_heur))]
average_recovery_heur = [store_results_heur[t][2] for t in range(len(store_results_heur))]

average_recovery_heur = [x for _, x in sorted(zip(average_transmissions_heur, average_recovery_heur))]
average_transmissions_heur.sort()

from BruteForceUtilityFunctions import lower_convex_hull

with open('Data/BruteForceIid.pickle', 'wb') as f:
    store_results_brute_force = pickle.load(f)

convex_hull_results = lower_convex_hull(store_results)
average_transmissions_lb = [convex_hull_results[t][0] for t in range(len(convex_hull_results))]
average_recovery_lb = [convex_hull_results[t][1] for t in range(len(convex_hull_results))]