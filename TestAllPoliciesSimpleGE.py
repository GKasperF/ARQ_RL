#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import Parallel, delayed
import multiprocessing
from BruteForceUtilityFunctions import ProbabilitySchedulingGE_Simple, ProbabilitySchedulingGE_simple_dec, lower_convex_hull, convex_hull
import pickle


# In[2]:


T = 20
Tf = 10
alpha = 0.25
beta = 0.25

num_cores = 6
store_results = Parallel(n_jobs = num_cores)(delayed(ProbabilitySchedulingGE_simple_dec)(alpha, beta, Tf, T, i) for i in range(1,2**(T+1)) )

with open('Data/BruteForceGESimple.pickle', 'wb') as f:
    pickle.dump(store_results, f)


# In[3]:


import matplotlib.pyplot as plt

average_transmissions = [store_results[t][0] for t in range(len(store_results))]
average_recovery = [store_results[t][1] for t in range(len(store_results))]

plt.plot(average_transmissions, average_recovery, 'x')
plt.show()


# In[40]:


import pickle

with open('BruteForceAllPoliciesGESimple.pickle', 'rb') as f:
    store_results = pickle.load(f)

convex_hull_results = lower_convex_hull(store_results)
i = 0

while 1:
    point = convex_hull_results[i]
    if point[0] < (1/(0.5) - 0.05):
        convex_hull_results.pop(i)
    else:
        i+=1
    
    if i == len(convex_hull_results):
        break
   
average_transmissions_lb = [convex_hull_results[t][0] for t in range(len(convex_hull_results))]
average_recovery_lb = [convex_hull_results[t][1] for t in range(len(convex_hull_results))]

import pickle

with open('StoredResults_SimpleGE_TestNumber3.pickle', 'rb') as f:
    results_policies = pickle.load(f)
    
average_transmissions_RL = [results_policies[t][1] for t in range(len(results_policies))]
average_recovery_RL = [results_policies[t][2] for t in range(len(results_policies))]

average_recovery_RL = [x for _, x in sorted(zip(average_transmissions_RL, average_recovery_RL))]
average_transmissions_RL.sort()


with open('Results_Heuristics_Simple_GE.pickle', 'rb') as f:
    store_results_heur = pickle.load(f)
                                    
convex_hull_heur = convex_hull(store_results_heur)
        
average_transmissions_heur = [convex_hull_heur[t][1] for t in range(len(convex_hull_heur))]
average_recovery_heur = [convex_hull_heur[t][2] for t in range(len(convex_hull_heur))]

average_recovery_heur = [x for _, x in sorted(zip(average_transmissions_heur, average_recovery_heur))]
average_transmissions_heur.sort()

plt.plot(average_transmissions_lb, average_recovery_lb, '-or', average_transmissions_RL, average_recovery_RL, 'x', average_transmissions_heur, average_recovery_heur, 'sg-' )
# plt.show()
plt.ylabel('Recovery Time')
plt.xlabel('Expected Transmissions')
plt.savefig('LowerBoundxRLxHeur_GE.pdf')
#AverageTrans.pickle


# In[25]:


import mpld3
mpld3.enable_notebook()

plt.plot(average_transmissions_RL, average_recovery_RL, 'x')
plt.show()

