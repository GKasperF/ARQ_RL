import torch
import copy
import Envs.PytorchEnvironments as EnvsNN
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def GenerateEpisodes(env, num_episodes):
    
    device = env.device

    states = torch.tensor([]).to(device)
    next_states = torch.tensor([]).to(device)
    rewards = torch.tensor([]).to(device)
    action_index = torch.tensor([]).to(device)
    actions = torch.tensor([]).to(device)

    action_probabilities = 0.5 * torch.ones((env.batch, 1)).to(device)
    num_finished_episodes = 0
    state = env.reset()
    # For every episode
    while num_finished_episodes < num_episodes:
        # get probabilities of all actions from current state
        action_index = torch.bernoulli(action_probabilities) 
        actions = torch.cat((actions, action_index), dim = 0)
        states = torch.cat((states, copy.deepcopy(state)), dim = 0)

        state, reward, done, SuccessF = env.step(action_index)
        next_states = torch.cat((next_states, copy.deepcopy(state)), dim = 0)
        rewards = torch.cat((rewards, reward))

        num_finished_episodes += torch.sum(SuccessF)

        state = env.reset_success()
    return(states, actions, rewards, next_states)

alpha_reward = 0
beta_reward = 5
Tf = 10
Nit = 100000
M = 5
batches = 1000
Channel = EnvsNN.GilbertElliott(0.25, 0.25, 0, 1, batches).to(device)
TransEnv = EnvsNN.EnvFeedbackGeneral(10, alpha_reward, 5, Channel, batches, M).to(device)
dataset = GenerateEpisodes(TransEnv, Nit)

with open('Data/DatasetGilbertElliott.pickle', 'wb') as f:
    pickle.dump(dataset, f)