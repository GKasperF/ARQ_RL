import numpy as np
from gym.utils import seeding
from joblib import Parallel, delayed
import multiprocessing
import pickle
import Envs.Environments as Envs
from tqdm import tqdm

def TestHeuristic(N_trans, N_zeros):
    #Channel_Local = Envs.GilbertElliott(0.25, 0.25, 0, 1)
    Channel_Local = Envs.GilbertElliott(0.1, 0.25, 0.05, 1)
    #Channel_Local = Envs.Fritchman(0.1, 0.5, 0.05, 3)
    #Channel_Local = Envs.iidchannel(0.1)
    TransEnvTest = Envs.EnvFeedbackGeneral(10, 1.4, 5, Channel_Local)
    policy_table = np.append(np.zeros(N_trans), np.ones(N_zeros))
    policy_table = policy_table.astype(int)
    reward_save2 = np.zeros((100000, 4))
    for i in range(100000):
        done = 0
        state = TransEnvTest.reset()
        reward_acc = 0
        transmissions = 0
        time_instant = 0
        number_successes = 0
        t = 0
        while 1:
          action = policy_table[t]
          if action == 0:
            transmissions += 1
          next_state, reward, done, SuccessF = TransEnvTest.step(TransEnvTest.actions[action])
          time_instant += 1
          reward_acc += reward
          state = next_state
          t += 1
          t = np.mod(t, len(policy_table))
          if done:
            if SuccessF:
              number_successes += 1
            break


        reward_save2[i][0] = reward_acc
        reward_save2[i][1] = transmissions
        reward_save2[i][2] = time_instant - TransEnvTest.Tf
        reward_save2[i][3] = number_successes

    average_reward_heur = (np.mean(reward_save2[:, 0]))
    average_transmissions_heur = (np.mean(reward_save2[:, 1]))
    average_recovery_heur = (np.mean(reward_save2[:, 2]))
    
    return(average_reward_heur, average_transmissions_heur, average_recovery_heur)

from joblib import Parallel, delayed
import multiprocessing

Tf = 10
num_cores = multiprocessing.cpu_count()

store_results_heur = Parallel(n_jobs = num_cores)(delayed(TestHeuristic)(N_trans, N_zeros) for N_trans in tqdm(range(1, Tf+1, 1)) for N_zeros in range(Tf - N_trans, Tf, 1))

with open('Data/HeuristicsResults_GE_Isolated_Example.pickle', 'wb') as f:
    pickle.dump(store_results_heur, f)