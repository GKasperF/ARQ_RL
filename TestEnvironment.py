import ReinforcementLearning.QlearningFunctions as QL
import Envs.PytorchEnvironments as EnvsNN
import torch
import numpy as np
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch = 10

Channel = EnvsNN.GilbertElliott(0.25, 0.25, 0, 1, batch).to(device)

#Channel = EnvsNN.Fritchman(0.25, 1, 0, 5, batch).to(device)
# output = torch.tensor([]).to(device)
# for i in range(10000):
#     temp = Channel.step()
#     output = torch.cat((output, temp.reshape((batch, 1)) ), dim=1)

# print(torch.mean(output))

env = EnvsNN.EnvFeedbackCheating_GE(10, 1.4, 5, Channel, batch)
env = env.to(device)

#actions = torch.ones(batch).to(device)
prob_actions = 0.5*torch.ones(batch).to(device)
while 1:
    actions = torch.bernoulli(prob_actions)
    state, reward, done, successF = env.step(actions)
    print(state)

# Qfunction = QL.QApproxFunction(env.observation_space.n, env.action_space.n, 1000).to(device)
# t0 = time.time()
# test = QL.GradientQLearningDebug(env, num_episodes= 1000, Qfunction = Qfunction, discount_factor = 0.95, epsilon = 1.0, UpdateEpisodes = 10)
# t1 = time.time()

#print('Training took {} seconds'.format(t1 - t0))

