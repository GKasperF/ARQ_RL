import pickle
import torch
from LowerBound.BruteForceUtilityFunctions import lower_convex_hull
import io 

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

store_results = []

with open('Data/AgentCNNRLresults_Iid_Example_RNN.pickle', 'rb') as f:
    while 1:
        try:
            test = CPU_Unpickler(f).load()
            temp1 = test[0].numpy()
            temp2 = test[1].numpy()
            temp3 = test[2].numpy()
            # temp1 = [test[t][0].numpy() for t in range(len(test))]
            # temp2 = [test[t][1].numpy() for t in range(len(test))]
            # temp3 = [test[t][2].numpy() for t in range(len(test))]
            store_results.append( (temp1, temp2, temp3)  )
        except (EOFError, pickle.UnpicklingError):
            break

with open('Data/AgentCNNRLresults_Iid_Example_RNN2.pickle', 'rb') as f:
    while 1:
        try:
            test = CPU_Unpickler(f).load()
            temp1 = test[0].numpy()
            temp2 = test[1].numpy()
            temp3 = test[2].numpy()
            # temp1 = [test[t][0].numpy() for t in range(len(test))]
            # temp2 = [test[t][1].numpy() for t in range(len(test))]
            # temp3 = [test[t][2].numpy() for t in range(len(test))]
            store_results.append( (temp1, temp2, temp3)  )
        except (EOFError, pickle.UnpicklingError):
            break

with open('Data/AgentCNNRLresults_Iid_Example_RNN3.pickle', 'wb') as f:
    pickle.dump(store_results, f)