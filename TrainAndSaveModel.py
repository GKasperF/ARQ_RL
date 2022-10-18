from time import time
import pickle
import dill
import ReinforcementLearning.QlearningFunctions as QL
import Envs.PytorchEnvironments as EnvsNN
import torch
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
params = sys.argv
if len(params) < 3:
    raise RuntimeError('Error: missing output file or alpha')
#params = ['nothing', 'TestNewTraining.pickle', '1.4']

output_file = 'Data/'+params[1]
alpha = float(params[2])

def TrainDebugNN(env, discount_factor, num_episodes, epsilon):
    Qfunction = QL.QApproxFunction(env.observation_space.n, env.action_space.n, 1000).to(env.device)
    lr_list = [0.001, 0.001, 0.001, 0.0001, 0.00001]
    for i in range(len(num_episodes)):
        Qfunction, policy, Debug = QL.GradientQLearningDebug(env, num_episodes[i], Qfunction, discount_factor, epsilon[i], UpdateEpisodes= 10, UpdateTargetEpisodes= 100, lr = lr_list[i])
        with open(output_file, 'wb') as f:
            pickle.dump(Qfunction, f)

    return(Qfunction, policy, Debug)

batches = 1000
Channel = EnvsNN.GilbertElliott(0.25, 0.25, 0, 1, batches).to(device)
TransEnv = EnvsNN.EnvFeedbackGeneral(10, alpha, 5, Channel, batches)
TransEnv = TransEnv.to(device)

num_episodes = [20000, 20000, 100000, 200000, 500000]
epsilon = [0.8, 0.6, 0.3, 0.2, 0.1]
discount_factor = 0.95

t0 = time()
QNN, policy, Debug = TrainDebugNN(TransEnv, discount_factor, num_episodes, epsilon)
t1 = time()

print('Training NN takes {} seconds'.format(t1-t0))

with open(output_file, 'wb') as f:
	pickle.dump(QNN, f)

with open('Data/SaveDebugCNN.pickle', 'wb') as f:
	pickle.dump(Debug, f)
