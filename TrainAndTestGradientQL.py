import numpy as np
import time
import copy
from joblib import Parallel, delayed
import multiprocessing
import pickle
import ReinforcementLearning.QlearningFunctions as QL
import Envs.PytorchEnvironments as Envs
import torch

q = []
if torch.cuda.is_available():
  num_cores = torch.cuda.device_count()
  for i in range(num_cores):
    q.append('cuda:'+'{}'.format(i))
else:
  num_cores = multiprocessing.cpu_count()
  for i in range(num_cores):
    q.append('cpu')

def TrainAndTest(alpha_reward, beta_reward, Tf, Nit, discount_factor, num_episodes, epsilon, batch, Channel):
    device = q.pop()
    Channel_Local = copy.deepcopy(Channel).to(device)
    string_alpha = str(alpha_reward.numpy())
    alpha_reward = alpha_reward.to(device)
    TransEnv = Envs.EnvFeedbackGeneral(Tf, alpha_reward, beta_reward, Channel_Local, batch, M=5)
    TransEnv = TransEnv.to(device)
    model_file = 'ModelCNNBatch_FritchmanBurstOnly'+string_alpha+'.pickle'
    t0 = time.time()
    Q, policy = Train(TransEnv, discount_factor, num_episodes, epsilon)
    t1 = time.time()
    
    with open('Data/'+model_file, 'wb') as f:
      pickle.dump(Q, f)    

    print('Training takes {} seconds'.format(t1 - t0))
    t0 = time.time()
    result = Test(TransEnv, Q, Nit, batch)
    t1 = time.time()
    print('Testing takes {} seconds'.format(t1 - t0))
    q.append(device)

    with open('Data/AgentCNNRLresultsTestBatch_FritchmanBurstOnly.pickle', 'ab') as f:
      pickle.dump(result, f)

    return(result)

def Train(env, discount_factor, num_episodes, epsilon):
    Qfunction = QL.QApproxFunction(env.observation_space.n, env.action_space.n, 1000).to(env.device)
    lr_list = [0.001, 0.001, 0.001, 0.0001, 0.00001]
    for i in range(len(num_episodes)):
        Qfunction, policy, _ = QL.GradientQLearningDebug(env, num_episodes[i], Qfunction, discount_factor, epsilon[i], UpdateEpisodes= 10, UpdateTargetEpisodes= 100, lr = lr_list[i])
    
    return(Qfunction, policy)

def Test(env, Q, Nit, batch):
    device = env.device
    reward_acc = torch.zeros(batch).to(device)
    transmissions = torch.zeros(batch).to(device)
    time_instant = torch.zeros(batch).to(device)
    number_successes = torch.zeros(batch).to(device)

    reward_save = torch.empty((0, 4)).to(device)
    for i in range(int(Nit/batch)):
        done = 0
        reward_acc[:] = 0
        transmissions[:] = 0
        time_instant[:] = 1
        number_successes[:] = 0
        state = env.reset()
        SuccessF = torch.zeros(batch).to(device)
        while 1:
          action_index = torch.argmax(Q(state), dim = 1)
          # take action and get reward, transit to next state
          transmissions[torch.logical_not(SuccessF)] = transmissions[torch.logical_not(SuccessF)] + action_index.reshape(len(action_index))[torch.logical_not(SuccessF)]

          next_state, reward, done, SuccessF = env.step(action_index)


          # Update statistics
          reward_acc += reward.reshape(len(reward))
          time_instant[ torch.logical_not(SuccessF)] += 1
          state = next_state
          if torch.any(time_instant > env.Tf) and torch.any(transmissions == 0):
            print('Learned bad policy')
            break
          if done:
            break
        
        temp = torch.cat( (reward_acc.reshape(batch, 1), transmissions.reshape(batch, 1), time_instant.reshape(batch, 1), number_successes.reshape(batch, 1)), dim = 1)
        reward_save = torch.cat( (reward_save, copy.deepcopy(temp)), dim = 0)
        
    average_reward = torch.mean(reward_save[:, 0])
    average_transmissions = torch.mean(reward_save[:, 1])
    average_recovery = torch.mean(reward_save[:, 2]) - env.Tf

    print('Estimated expected reward is {} \n Expected reward is: {}'.format(Q(env.reset()), average_reward))
    
    return(average_reward, average_transmissions, average_recovery)


alpha_range = torch.arange(0.1, 5.5, 0.1)
#alpha_range = torch.tensor([0.1, 0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0])
beta_reward = 5
Tf = 10
Nit = 10000
epsilon = [0.8, 0.6, 0.3, 0.2, 0.1]
discount_factor = 0.95

batches = 100

#Channel = Envs.GilbertElliott(0.25, 0.25, 0, 1, batches)
Channel = Envs.Fritchman(0.25, 1, 0, 5, batches)

num_episodes = [int(2000), int(2000), int(10000), int(20000), int(50000)]

store_results = Parallel(n_jobs = num_cores, require='sharedmem')(delayed(TrainAndTest)(alpha_reward, beta_reward, Tf, Nit, discount_factor, num_episodes, epsilon, batches, Channel) for alpha_reward in alpha_range)
#store_results = TrainAndTest(alpha_range[0], beta_reward, Tf, Nit, discount_factor, num_episodes, epsilon, batches, Channel)
with open('Data/AgentCNNRLresultsTestBatch_FritchmanBurstOnly.pickle', 'wb') as f:
    pickle.dump(store_results, f)