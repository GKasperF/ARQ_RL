#!/usr/bin/env python
# coding: utf-8

import itertools
import numpy as np
import copy
import torch
import torch.nn.functional as F

def createEpsilonGreedyPolicyGradient(Q, epsilon, num_actions, batch = 1):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.
      
    Returns a function that takes the state
    as an input and returns the probabilities
    for each action in the form of a numpy array 
    of length of the action space(set of possible actions).
    """
    def policyFunction(state):
        Action_probabilities = epsilon/num_actions * torch.ones((batch, num_actions), dtype = float)
        best_action = torch.argmax(Q(state), dim = 1)
        Action_probabilities[range(batch), best_action] += (1.0 - epsilon)
        return Action_probabilities
  
    return policyFunction

class QApproxFunction(torch.nn.Module):
  def __init__(self, state_dim: 'Number of state variables', action_dim: 'Number of possible actions', hidden_layer: 'Size of the hidden layers'):
    super(QApproxFunction, self).__init__()

    self.state_dim = state_dim
    self.action_dim = action_dim
    self.hidden_layer = hidden_layer
    
    self.Layer1 = torch.nn.Linear(state_dim, 1000*state_dim)
    # self.Layer2 = torch.nn.Linear(1000, 500)
    # self.Layer3 = torch.nn.Linear(500, 250)
    # self.Layer4 = torch.nn.Linear(250, 100)
    # self.Layer5 = torch.nn.Linear(100, 50)

    self.Layer2 = torch.nn.Conv1d(hidden_layer, hidden_layer, 3, padding = 1)
    self.Layer3 = torch.nn.Conv1d(hidden_layer, hidden_layer, 3, padding = 1)
    self.Layer4 = torch.nn.Conv1d(hidden_layer, hidden_layer, 3, padding = 1)
    self.Layer5 = torch.nn.Conv1d(hidden_layer, hidden_layer, 3, padding = 1)

    self.Layer6 = torch.nn.Conv1d(hidden_layer, 1, 1)
    
    self.FinalLayer = torch.nn.Linear(state_dim, action_dim)

  def forward(self, x):
    L1 = self.Layer1(x)
    L1_reshaped = L1.view(-1, self.hidden_layer, self.state_dim)
    ReLU1 = F.relu(L1_reshaped)
    L2 = self.Layer2(ReLU1)
    ReLU2 = F.relu(L2)
    L3 = self.Layer3(ReLU2)
    ReLU3 = F.relu(L3)
    L4 = self.Layer4(ReLU3)
    ReLU4 = F.relu(L4)
    L5 = self.Layer5(ReLU4)
    ReLU5 = F.relu(L5)
    L6 = self.Layer6(ReLU5)

    
    output = self.FinalLayer(L6.view(-1, self.state_dim))
    
    return output

def GradientQLearning(env, num_episodes, Qfunction , discount_factor = 1.0,
                            epsilon = 0.1, UpdateEpisodes = 10, UpdateTargetEpisodes = 100, lr = 0.001):
    device = env.device
    
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(list(Qfunction.parameters()), lr = lr)

    states = torch.tensor([]).to(device)
    next_states = torch.tensor([]).to(device)
    rewards = torch.tensor([]).to(device)
    action_index = torch.tensor([]).to(device)
    actions = torch.tensor([]).to(device)

    Probability_Basis = epsilon/env.action_space.n * torch.ones((env.batch, 1)).to(device)
    Sum_Probability = (1.0 - epsilon)*torch.ones((env.batch), 1).to(device).squeeze()
    Zeros_Tensor = torch.zeros((env.batch, 1)).to(device).squeeze()
    num_finished_episodes = 0
    state = env.reset()
    Qtarget = copy.deepcopy(Qfunction)
    num_successes = torch.zeros(1).to(device)
    count0 = 0
    rewards_acc = torch.zeros((env.batch, 1)).to(device)
    # For every episode
    while num_finished_episodes < num_episodes:
        # get probabilities of all actions from current state
        best_action = torch.argmax(Qfunction(state), dim = 1)
        action_probabilities = Probability_Basis + torch.where(best_action == 1, Sum_Probability, Zeros_Tensor).reshape(Probability_Basis.shape)
        action_index = torch.bernoulli(action_probabilities[:, 0]) 
        actions = torch.cat((actions, action_index), dim = 0)

        #states.append(copy.deepcopy(state))
        states = torch.cat((states, copy.deepcopy(state)), dim = 0)

        # take action and get reward, transit to next state
        state, reward, done, SuccessF = env.step(action_index)
        next_states = torch.cat((next_states, copy.deepcopy(state)), dim = 0)
        rewards = torch.cat((rewards, reward))
        
        rewards_acc += reward.reshape(rewards_acc.shape)
        if torch.sum(SuccessF) > 0:
            rewards_acc[SuccessF==1] = 0

        num_successes += torch.sum(SuccessF)
        count0 += torch.sum(SuccessF)
        num_finished_episodes += torch.sum(SuccessF)

        state = env.reset_success()

        if num_successes > UpdateEpisodes:
            Next_States_QValues = Qtarget(next_states)
            finish_state = env.finish_state[0]
            finish_states_indices = torch.all(torch.eq(next_states, finish_state), dim = 1)
            finish_states_indices = finish_states_indices.reshape( len(finish_states_indices), 1).repeat(1,env.action_space.n)
            Next_States_QValues = torch.where(finish_states_indices, torch.zeros(Next_States_QValues.size()).to(device), Next_States_QValues)

            BestTargetValues, _ = torch.max(Next_States_QValues, dim = 1, keepdim = True)
            td_target = rewards.reshape((len(rewards), 1)) + discount_factor*BestTargetValues
            td_target = td_target.detach()
            for i in range(num_successes.type(torch.int64)):
                Qestimates = Qfunction(states)
                td_estimate = Qestimates[torch.arange(len(states)), actions.type(torch.int64)].reshape( (len(states), 1))
                loss = criterion(td_estimate, td_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            states = states[0:0]
            next_states = next_states[0:0]
            rewards = rewards[0:0]
            actions = actions[0:0]
            num_successes = 0

        if count0 > UpdateTargetEpisodes:
            Qtarget = copy.deepcopy(Qfunction)
            count0 = 0

        
    if len(states) > 0:    
        Next_States_QValues = Qtarget(next_states)
        finish_state = env.finish_state[0]
        finish_states_indices = torch.all(torch.eq(next_states, finish_state), dim = 1)
        finish_states_indices = finish_states_indices.reshape( len(finish_states_indices), 1).repeat(1,env.action_space.n)
        Next_States_QValues = torch.where(finish_states_indices, torch.zeros(Next_States_QValues.size()).to(device), Next_States_QValues)

        BestTargetValues, _ = torch.max(Next_States_QValues, dim = 1, keepdim = True)
        td_target = rewards.reshape((len(rewards), 1)) + discount_factor*BestTargetValues
        Qestimates = Qfunction(states)
        td_estimate = Qestimates[torch.arange(len(states)), actions.type(torch.int64)].reshape( (len(states), 1))
        loss = criterion(td_estimate, td_target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    policy = createEpsilonGreedyPolicyGradient(Qfunction, 0, env.action_space.n)
       
    return Qfunction, policy

def GradientQLearningDebug(env, num_episodes, Qfunction , discount_factor = 1.0,
                            epsilon = 0.1, UpdateEpisodes = 10, UpdateTargetEpisodes = 100, lr = 0.001):
    
    device = env.device
    
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(list(Qfunction.parameters()), lr = lr)

    states = torch.tensor([]).to(device)
    next_states = torch.tensor([]).to(device)
    rewards = torch.tensor([]).to(device)
    action_index = torch.tensor([]).to(device)
    actions = torch.tensor([]).to(device)

    Debug1 = torch.tensor([]).to(device)
    state_of_interest = torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0]]).to(device)
    Debug2 = torch.tensor([]).to(device)

    Probability_Basis = epsilon/env.action_space.n * torch.ones((env.batch, 1)).to(device)
    Sum_Probability = (1.0 - epsilon)*torch.ones((env.batch), 1).to(device).squeeze()
    Zeros_Tensor = torch.zeros((env.batch, 1)).to(device).squeeze()
    num_finished_episodes = 0
    state = env.reset()
    Qtarget = copy.deepcopy(Qfunction)
    num_successes = torch.zeros(1).to(device)
    count0 = 0
    rewards_acc = torch.zeros((env.batch, 1)).to(device)
    # For every episode
    while num_finished_episodes < num_episodes:
        # get probabilities of all actions from current state
        best_action = torch.argmax(Qfunction(state), dim = 1)
        action_probabilities = Probability_Basis + torch.where(best_action == 1, Sum_Probability, Zeros_Tensor).reshape(Probability_Basis.shape)
        action_index = torch.bernoulli(action_probabilities[:, 0]) 
        actions = torch.cat((actions, action_index), dim = 0)

        #states.append(copy.deepcopy(state))
        states = torch.cat((states, copy.deepcopy(state)), dim = 0)

        # take action and get reward, transit to next state
        state, reward, done, SuccessF = env.step(action_index)
        next_states = torch.cat((next_states, copy.deepcopy(state)), dim = 0)
        rewards = torch.cat((rewards, reward))
        
        rewards_acc += reward.reshape(rewards_acc.shape)
        if torch.sum(SuccessF) > 0:
            Debug2 = torch.cat((Debug2, torch.mean(rewards_acc[SuccessF==1]).reshape(1) ))
            rewards_acc[SuccessF==1] = 0

        num_successes += torch.sum(SuccessF)
        count0 += torch.sum(SuccessF)
        num_finished_episodes += torch.sum(SuccessF)

        state = env.reset_success()

        if num_successes > UpdateEpisodes:
            Next_States_QValues = Qtarget(next_states)
            finish_state = env.finish_state[0]
            finish_states_indices = torch.all(torch.eq(next_states, finish_state), dim = 1)
            finish_states_indices = finish_states_indices.reshape( len(finish_states_indices), 1).repeat(1,env.action_space.n)
            Next_States_QValues = torch.where(finish_states_indices, torch.zeros(Next_States_QValues.size()).to(device), Next_States_QValues)

            BestTargetValues, _ = torch.max(Next_States_QValues, dim = 1, keepdim = True)
            td_target = rewards.reshape((len(rewards), 1)) + discount_factor*BestTargetValues
            td_target = td_target.detach()
            for i in range(num_successes.type(torch.int64)):
                Qestimates = Qfunction(states)
                td_estimate = Qestimates[torch.arange(len(states)), actions.type(torch.int64)].reshape( (len(states), 1))
                loss = criterion(td_estimate, td_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            states = states[0:0]
            next_states = next_states[0:0]
            rewards = rewards[0:0]
            actions = actions[0:0]
            num_successes = 0

            Debug1 = torch.cat((Debug1, loss.reshape(1,1)))

        if count0 > UpdateTargetEpisodes:
            Qtarget = copy.deepcopy(Qfunction)
            count0 = 0

        
    if len(states) > 0:    
        Next_States_QValues = Qtarget(next_states)
        finish_state = env.finish_state[0]
        finish_states_indices = torch.all(torch.eq(next_states, finish_state), dim = 1)
        finish_states_indices = finish_states_indices.reshape( len(finish_states_indices), 1).repeat(1,env.action_space.n)
        Next_States_QValues = torch.where(finish_states_indices, torch.zeros(Next_States_QValues.size()).to(device), Next_States_QValues)

        BestTargetValues, _ = torch.max(Next_States_QValues, dim = 1, keepdim = True)
        td_target = rewards.reshape((len(rewards), 1)) + discount_factor*BestTargetValues
        Qestimates = Qfunction(states)
        td_estimate = Qestimates[torch.arange(len(states)), actions.type(torch.int64)].reshape( (len(states), 1))
        loss = criterion(td_estimate, td_target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    Debug1 = torch.cat((Debug1, loss.reshape(1,1)))
    #Debug2 = torch.cat((Debug2, Qfunction(state_of_interest)))        

    policy = createEpsilonGreedyPolicyGradient(Qfunction, 0, env.action_space.n)
    Debug1 = Debug1.detach().to('cpu').numpy()
    Debug2 = Debug2.detach().to('cpu').numpy()
    Debug = [Debug1, Debug2]
       
    return Qfunction, policy, Debug