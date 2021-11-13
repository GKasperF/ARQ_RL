from time import time
import pickle
import dill
import ReinforcementLearning.QlearningFunctions as QL
import Envs.PytorchEnvironments as EnvsNN
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def TrainDebugNN(env, discount_factor, num_episodes):
    Qfunction = QL.QApproxFunction(env.observation_space.n, env.action_space.n, 1000).to(device)
    for i in range(len(num_episodes)):
        Qfunction, policy = QL.GradientRandomQLearning(env, num_episodes[i], Qfunction , discount_factor, UpdateEpisodes=10)
    
    return(Qfunction, policy)

batches = 8
Channel = EnvsNN.GilbertElliott(0.25, 0.25, 0, 1)
TransEnv = EnvsNN.EnvFeedbackGeneral(10, 1.4, 5, Channel, batches)
TransEnv = TransEnv.to(device)

num_episodes = [1000]
discount_factor = 0.95

t0 = time()
QNN, policy = TrainDebugNN(TransEnv, discount_factor, num_episodes)
t1 = time()

print('Training NN takes {} seconds'.format(t1-t0))

with open('Data/SaveModelCNNRandomTraining.pickle', 'wb') as f:
	pickle.dump(QNN, f)

# with open('Data/SaveDebugCNNRandomTraining.pickle', 'wb') as f:
# 	pickle.dump(Debug, f)
