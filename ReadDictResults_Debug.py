import torch
import pickle
import io

class CPU_Unpickler(pickle.Unpickler):
  def find_class(self, module, name):
      if module == 'torch.storage' and name == '_load_from_bytes':
          return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
      else: return super().find_class(module, name)

with open('Data/AgentCNNRLresults_Iid_Example_RNN_Dict.pickle', 'rb') as f:
    result_dict = CPU_Unpickler(f).load()

pass
print('ok')