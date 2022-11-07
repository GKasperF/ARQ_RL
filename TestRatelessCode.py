import numpy as np
from gym.utils import seeding
from joblib import Parallel, delayed
import multiprocessing
import pickle
import Envs.Environments as Envs
from tqdm import tqdm

def TestRateless(PacketSize, TotalPackets):
    Channel_Local = Envs.GilbertElliott(0.25, 0.25, 0, 1)
    #Channel_Local = Envs.GilbertElliott(0.1, 0.25, 0.05, 1)
    #Channel_Local = Envs.Fritchman(0.1, 0.5, 0.05, 3)
    #Channel_Local = Envs.iidchannel(0.1)
    
    
    time_received = np.inf*np.ones((TotalPackets, 1))

    num_dofs = 0
    num_vars = 0
    last_received = -1

    t = 0

    while np.isinf(time_received[-1]):
        erasure = Channel_Local.step()
        if t < TotalPackets:
          num_vars = num_vars + 1

        if erasure == 0:
          num_dofs = num_dofs + PacketSize
        
        if num_dofs > num_vars:
          time_received[last_received+1 : t + 1] = t
          last_received = t
          num_dofs = 0
          num_vars = 0

        t = t + 1

    Num_Transmissions = t
    delay = np.zeros((TotalPackets, 1))
    for t in range(TotalPackets):
      delay[t] = time_received[t] - t

    average_transmissions_heur = (np.array(PacketSize) * Num_Transmissions / TotalPackets)
    average_recovery_heur = (np.mean(delay))
    
    return(average_transmissions_heur, average_recovery_heur)

from joblib import Parallel, delayed
import multiprocessing

Tf = 10
num_cores = multiprocessing.cpu_count()
TotalPackets = 100000

store_results_heur = []
# for PacketSize in tqdm(np.arange(1, 5, 0.5)):
#   store_results_heur.append(TestRateless(PacketSize, TotalPackets))

store_results_heur = Parallel(n_jobs = num_cores)(delayed(TestRateless)(PacketSize, TotalPackets) for PacketSize in tqdm(np.arange(1, 12, 0.5)))

print(store_results_heur)

with open('Data/RatelessResults_GE_Isolated_Example.pickle', 'wb') as f:
    pickle.dump(store_results_heur, f)