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
                            epsilon = 0.1, UpdateEpisodes=10):
    device = env.device
    
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(list(Qfunction.parameters()))

    # For every episode
    for ith_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        policy = createEpsilonGreedyPolicyGradient(Qfunction, epsilon, env.action_space.n, env.batch)
        states = torch.tensor([]).to(device)
        next_states = torch.tensor([]).to(device)
        actions = []
        rewards = torch.tensor([]).to(device)

        if ith_episode % UpdateEpisodes == 0:
            Qtarget = copy.deepcopy(Qfunction)

        for t in itertools.count():
            # get probabilities of all actions from current state
            action_probabilities = policy(state)   
            action_index = []
            for i in range(env.batch):
                action_temp = np.random.choice( np.arange( len(action_probabilities[i])), p = action_probabilities[i])
                action_index.append(env.actions[action_temp])
                actions.append(action_temp)

            #states.append(copy.deepcopy(state))
            states = torch.cat((states, copy.deepcopy(state)), dim = 0)

            # take action and get reward, transit to next state
            state, reward, done, SuccessF = env.step(action_index)
            next_states = torch.cat((next_states, copy.deepcopy(state)), dim = 0)
            rewards = torch.cat((rewards, reward))
            
            if done:
                break

        
        Next_States_QValues = Qtarget(next_states)
        finish_state = env.finish_state[0]
        finish_states_indices = torch.all(torch.eq(next_states, finish_state), dim = 1)
        finish_states_indices = finish_states_indices.reshape( len(finish_states_indices), 1).repeat(1,env.action_space.n)
        Next_States_QValues = torch.where(finish_states_indices, torch.zeros(Next_States_QValues.size()).to(device) , Next_States_QValues)

        BestTargetValues, _ = torch.max(Next_States_QValues, dim = 1, keepdim = True)
        td_target = rewards.reshape((len(rewards), 1)) + discount_factor*BestTargetValues

        Qestimates = Qfunction(states)
        unfinished_states_indices = torch.logical_not(torch.all(torch.eq(states, finish_state), dim = 1))

        td_target = td_target[unfinished_states_indices]
        td_estimate = Qestimates[torch.arange(len(states)), actions].reshape( (len(states), 1))
        td_estimate = td_estimate[unfinished_states_indices]

        loss = criterion(td_estimate, td_target.detach())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    policy = createEpsilonGreedyPolicyGradient(Qfunction, 0, env.action_space.n)
       
    return Qfunction, policy

def GradientQLearningDebug(env, num_episodes, Qfunction , discount_factor = 1.0,
                            epsilon = 0.1, UpdateEpisodes = 10, lr = 0.001):
    
    device = env.device
    
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(list(Qfunction.parameters()), lr = lr)
    
    Debug = []

    states = torch.tensor([]).to(device)
    next_states = torch.tensor([]).to(device)
    rewards = torch.tensor([]).to(device)
    action_index = torch.tensor([]).to(device)
    actions = torch.tensor([]).to(device)

    Probability_Basis = epsilon/env.action_space.n * torch.ones((env.batch, 1)).to(device)
    Sum_Probability = (1.0 - epsilon)*torch.ones((env.batch), 1).to(device)
    Zeros_Tensor = torch.zeros((env.batch, 1)).to(device)

    # For every episode
    for ith_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        states = states[0:0]
        next_states = next_states[0:0]
        rewards = rewards[0:0]
        actions = actions[0:0]
        if ith_episode % UpdateEpisodes == 0:
            Qtarget = copy.deepcopy(Qfunction)

        for t in itertools.count():
            # get probabilities of all actions from current state
            best_action = torch.argmax(Qfunction(state), dim = 1)
            action_probabilities = Probability_Basis + torch.where(best_action == 1, Sum_Probability.squeeze(), Zeros_Tensor.squeeze()).reshape(Probability_Basis.shape)
            action_index = torch.bernoulli(action_probabilities[:, 0]) 
            actions = torch.cat((actions, action_index), dim = 0)

            #states.append(copy.deepcopy(state))
            states = torch.cat((states, copy.deepcopy(state)), dim = 0)

            # take action and get reward, transit to next state
            state, reward, done, SuccessF = env.step(action_index)
            next_states = torch.cat((next_states, copy.deepcopy(state)), dim = 0)
            rewards = torch.cat((rewards, reward))
            
            if done:
                break

        
        Next_States_QValues = Qtarget(next_states)
        finish_state = env.finish_state[0]
        finish_states_indices = torch.all(torch.eq(next_states, finish_state), dim = 1)
        finish_states_indices = finish_states_indices.reshape( len(finish_states_indices), 1).repeat(1,env.action_space.n)
        Next_States_QValues = torch.where(finish_states_indices, torch.zeros(Next_States_QValues.size()).to(device) , Next_States_QValues)

        BestTargetValues, _ = torch.max(Next_States_QValues, dim = 1, keepdim = True)
        td_target = rewards.reshape((len(rewards), 1)) + discount_factor*BestTargetValues

        Qestimates = Qfunction(states)
        unfinished_states_indices = torch.logical_not(torch.all(torch.eq(states, finish_state), dim = 1))

        td_target = td_target[unfinished_states_indices]
        td_estimate = Qestimates[torch.arange(len(states)), actions.type(torch.int64)].reshape( (len(states), 1))
        td_estimate = td_estimate[unfinished_states_indices]

        loss = criterion(td_estimate, td_target.detach())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    policy = createEpsilonGreedyPolicyGradient(Qfunction, 0, env.action_space.n)
       
    return Qfunction, policy, Debug

def GradientRandomQLearning(env, num_episodes, Qfunction , discount_factor = 1.0, UpdateEpisodes=10, lr = 0.001):
    device = env.device
    
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(list(Qfunction.parameters()), lr = lr)
    action_probabilities = 0.5 * torch.ones((env.batch,1)).to(device)
    states = torch.tensor([]).to(device)
    next_states = torch.tensor([]).to(device)
    rewards = torch.tensor([]).to(device)
    action_index = torch.tensor([]).to(device)
    actions = torch.tensor([]).to(device)
    # For every episode
    for ith_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        states = states[0:0]
        next_states = next_states[0:0]
        rewards = rewards[0:0]
        actions = actions[0:0]
        if ith_episode % UpdateEpisodes == 0:
            Qtarget = copy.deepcopy(Qfunction)
        
        for t in itertools.count():
            # get probabilities of all actions from current state
               
            action_index = torch.bernoulli(action_probabilities)
            actions = torch.cat((actions, action_index), dim = 0)
            #states.append(copy.deepcopy(state))
            states = torch.cat((states, copy.deepcopy(state)), dim = 0)

            # take action and get reward, transit to next state
            state, reward, done, SuccessF = env.step(action_index)
            next_states = torch.cat((next_states, copy.deepcopy(state)), dim = 0)
            rewards = torch.cat((rewards, reward))
            
            if done:
                break

        
        Next_States_QValues = Qtarget(next_states)
        finish_state = env.finish_state[0]
        finish_states_indices = torch.all(torch.eq(next_states, finish_state), dim = 1)
        finish_states_indices = finish_states_indices.reshape( len(finish_states_indices), 1).repeat(1,env.action_space.n)
        Next_States_QValues = torch.where(finish_states_indices, torch.zeros(Next_States_QValues.size()).to(device) , Next_States_QValues)

        BestTargetValues, _ = torch.max(Next_States_QValues, dim = 1, keepdim = True)
        td_target = rewards.reshape((len(rewards), 1)) + discount_factor*BestTargetValues

        Qestimates = Qfunction(states)
        unfinished_states_indices = torch.logical_not(torch.all(torch.eq(states, finish_state), dim = 1))

        td_target = td_target[unfinished_states_indices]
        td_estimate = Qestimates[torch.arange(len(states)), actions.type(torch.int64).reshape(len(actions))].reshape( (len(states), 1))
        td_estimate = td_estimate[unfinished_states_indices]

        loss = criterion(td_estimate, td_target.detach())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    policy = createEpsilonGreedyPolicyGradient(Qfunction, 0, env.action_space.n)
       
    return Qfunction, policy

def GradientQLearningMC(env, num_episodes, Qfunction, epsilon = 0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""
       
    #Qfunction = QApproxFunction(env.observation_space.n, env.action_space.n)
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = env.device
    #device = 'cpu'
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(Qfunction.parameters()))
    # For every episode
    for ith_episode in range(num_episodes):
        # Reset the environment
        state = env.reset()
        policy = createEpsilonGreedyPolicyGradient(Qfunction, epsilon, env.action_space.n) 
        states = []
        actions = []
        rewards = torch.tensor([]).to(device)
        for t in itertools.count():
            # get probabilities of all actions from current state
            action_probabilities = policy(state)
            # choose action according to 
            # the probability distribution
            action_index = np.random.choice(np.arange(
                      len(action_probabilities)),
                       p = action_probabilities)
            states.append(copy.deepcopy(state))
            actions.append(action_index)
            # take action and get reward, transit to next state
            state, reward, done, SuccessF = env.step(env.actions[action_index])
            rewards = torch.cat((rewards, reward))
            if done:
                break

        for t in range(len(rewards)):
            target = torch.sum(rewards[t:], dim = 0, keepdim = True).detach()
            prediction = Qfunction(states[t])[:,actions[t]]
            loss = criterion(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    policy = createEpsilonGreedyPolicyGradient(Qfunction, 0, env.action_space.n)
       
    return Qfunction, policy

def GradientQLearningDebugMC(env, num_episodes, Qfunction, epsilon = 0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""
       
    #Qfunction = QApproxFunction(env.observation_space.n, env.action_space.n)
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = env.device
    #device = 'cpu'
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(Qfunction.parameters()))
    
    Debug = []
    # For every episode
    for ith_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        policy = createEpsilonGreedyPolicyGradient(Qfunction, epsilon, env.action_space.n) 
        #Predictions = torch.tensor([]).to(device)
        states = []
        actions = []
        rewards = torch.tensor([]).to(device)
        for t in itertools.count():
            # get probabilities of all actions from current state
            
            action_probabilities = policy(state)
   
            # choose action according to 
            # the probability distribution
            action_index = np.random.choice(np.arange(
                      len(action_probabilities)),
                       p = action_probabilities)
            #Predictions = torch.cat((Predictions, Qfunction(state)[:,action_index]))
            states.append(copy.deepcopy(state))
            actions.append(action_index)
            #Predictions.backward()
            # take action and get reward, transit to next state
            state, reward, done, SuccessF = env.step(env.actions[action_index])
            rewards = torch.cat((rewards, reward))
            if done:
                break

        for t in range(len(rewards)):
            target = torch.sum(rewards[t:], dim = 0, keepdim = True).detach()
            prediction = Qfunction(states[t])[:,actions[t]]
            loss = criterion(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Debug.append(loss.to('cpu').detach().numpy())

    policy = createEpsilonGreedyPolicyGradient(Qfunction, 0, env.action_space.n)
       
    return Qfunction, policy, Debug