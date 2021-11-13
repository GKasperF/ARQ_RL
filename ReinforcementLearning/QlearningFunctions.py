#!/usr/bin/env python
# coding: utf-8

import itertools
import numpy as np
import copy
import torch
import torch.nn.functional as F

def createEpsilonGreedyPolicyGradient(Q, epsilon, num_actions):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.
      
    Returns a function that takes the state
    as an input and returns the probabilities
    for each action in the form of a numpy array 
    of length of the action space(set of possible actions).
    """
    def policyFunction(state):
  
        Action_probabilities = np.ones(num_actions,
                dtype = float) * epsilon / num_actions
                  
        best_action = torch.argmax(Q(state), dim = 1)
        Action_probabilities[best_action] += (1.0 - epsilon)
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
        policy = createEpsilonGreedyPolicyGradient(Qfunction, epsilon, env.action_space.n)
        states = torch.tensor([]).to(device)
        actions = []
        rewards = torch.tensor([]).to(device)

        if ith_episode % UpdateEpisodes == 0:
            Qtarget = copy.deepcopy(Qfunction)

        for t in itertools.count():
            # get probabilities of all actions from current state
            action_probabilities = policy(state)
   
            # choose action according to 
            # the probability distribution
            action_index = np.random.choice(np.arange(
                      len(action_probabilities)),
                       p = action_probabilities)

            #states.append(copy.deepcopy(state))
            states = torch.cat((states, copy.deepcopy(state)), dim = 0)
            actions.append(action_index)

            # take action and get reward, transit to next state
            state, reward, done, SuccessF = env.step(env.actions[action_index])
            rewards = torch.cat((rewards, reward))
            
            if done:
                break


        TargetValues = torch.cat((Qtarget(states[1:]), torch.tensor([[0., 0.]]).to(device)), dim = 0)
        BestTargetValues, _ = torch.max(TargetValues, dim = 1, keepdim = True)
        td_target = rewards.reshape((len(rewards), 1)) + discount_factor*BestTargetValues
        td_estimate = Qfunction(states)[torch.arange(len(states)), actions].reshape( (len(states), 1))

        loss = criterion(td_estimate, td_target.detach())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    policy = createEpsilonGreedyPolicyGradient(Qfunction, 0, env.action_space.n)
       
    return Qfunction, policy

def GradientQLearningDebug(env, num_episodes, Qfunction , discount_factor = 1.0,
                            epsilon = 0.1, UpdateEpisodes = 10):
    
    device = env.device
    
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(list(Qfunction.parameters()))
    
    Debug = []

    # For every episode
    for ith_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        policy = createEpsilonGreedyPolicyGradient(Qfunction, epsilon, env.action_space.n)
        states = torch.tensor([]).to(device)
        actions = []
        rewards = torch.tensor([]).to(device)

        if ith_episode % UpdateEpisodes == 0:
            Qtarget = copy.deepcopy(Qfunction)

        for t in itertools.count():
            # get probabilities of all actions from current state
            action_probabilities = policy(state)
   
            # choose action according to 
            # the probability distribution
            action_index = np.random.choice(np.arange(
                      len(action_probabilities)),
                       p = action_probabilities)

            #states.append(copy.deepcopy(state))
            states = torch.cat((states, copy.deepcopy(state)), dim = 0)
            actions.append(action_index)

            # take action and get reward, transit to next state
            state, reward, done, SuccessF = env.step(env.actions[action_index])
            rewards = torch.cat((rewards, reward))
            
            if done:
                break


        TargetValues = torch.cat((Qtarget(states[1:]), torch.tensor([[0., 0.]]).to(device)), dim = 0)
        BestTargetValues, _ = torch.max(TargetValues, dim = 1, keepdim = True)
        td_target = rewards.reshape((len(rewards), 1)) + discount_factor*BestTargetValues
        td_estimate = Qfunction(states)[torch.arange(len(states)), actions].reshape( (len(states), 1))

        loss = criterion(td_estimate, td_target.detach())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    policy = createEpsilonGreedyPolicyGradient(Qfunction, 0, env.action_space.n)
       
    return Qfunction, policy, Debug

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