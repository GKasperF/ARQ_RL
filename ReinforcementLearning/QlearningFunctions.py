#!/usr/bin/env python
# coding: utf-8

# In[1]:

from collections import defaultdict
import itertools
import numpy as np
import copy
import torch
import torch.nn.functional as F

def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
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
                  
        best_action = np.argmax(Q[state.tobytes()])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities
  
    return policyFunction

class DebugStruct():
    def __init__(self, num_episodes):
      self.episode_state_value_send = np.zeros(num_episodes)
      self.episode_state_value_silence = np.zeros(num_episodes)

def qLearningDebug(env, num_episodes, Q_input, discount_factor = 1.0,
                            alpha = 0.6, epsilon = 0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""
       
    Q = copy.deepcopy(Q_input)
    NumberVisits = defaultdict(lambda: np.zeros(env.action_space.n))
    


    DebugStr = DebugStruct(num_episodes)
    state_of_interest = env.reset()
       
    # For every episode
    for ith_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)   
        for t in itertools.count():
               
            # get probabilities of all actions from current state
            action_probabilities = policy(state)
   
            # choose action according to 
            # the probability distribution
            action_index = np.random.choice(np.arange(
                      len(action_probabilities)),
                       p = action_probabilities)
            state = state.copy()
            # take action and get reward, transit to next state
            next_state, reward, done, SuccessF = env.step(env.actions[action_index])
   
            # TD Update
            next_state_index = next_state.tobytes()
            state_index = state.tobytes()

            best_next_action = np.argmax(Q[next_state_index])    
            td_target = reward + discount_factor * Q[next_state_index][best_next_action]
            td_delta = td_target - Q[state_index][action_index]
            Q[state_index][action_index] += alpha * td_delta
            NumberVisits[state_index][action_index] += 1
   
            # done is True if episode terminated   
            if done:
                DebugStr.episode_state_value_send[ith_episode] = Q[state_of_interest.tobytes()][0]
                DebugStr.episode_state_value_silence[ith_episode] = Q[state_of_interest.tobytes()][1]
                break
                   
            state = next_state

    policy = createEpsilonGreedyPolicy(Q, 0, env.action_space.n)
       
    return Q, policy, NumberVisits, DebugStr

def qLearning(env, num_episodes, Q_input, discount_factor = 1.0,
                            alpha = 0.6, epsilon = 0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""
       
    Q = copy.deepcopy(Q_input)
       
    # For every episode
    for ith_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)   
        for t in itertools.count():
               
            # get probabilities of all actions from current state
            action_probabilities = policy(state)
   
            # choose action according to 
            # the probability distribution
            action_index = np.random.choice(np.arange(
                      len(action_probabilities)),
                       p = action_probabilities)
            state = copy.deepcopy(state)
            # take action and get reward, transit to next state
            next_state, reward, done, SuccessF = env.step(env.actions[action_index])
               
            # TD Update
            next_state_index = next_state.tobytes()
            state_index = state.tobytes()

            best_next_action = np.argmax(Q[next_state_index])    
            td_target = reward + discount_factor * Q[next_state_index][best_next_action]
            td_delta = td_target - Q[state_index][action_index]
            Q[state_index][action_index] += alpha * td_delta
   
            # done is True if episode terminated   
            if done:
                break
                   
            state = next_state

    policy = createEpsilonGreedyPolicy(Q, 0, env.action_space.n)
       
    return Q, policy

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
  def __init__(self, state_dim: 'Number of state variables', action_dim: 'Number of possible actions'):
    super(QApproxFunction, self).__init__()

    self.state_dim = state_dim
    self.action_dim = action_dim
    
    self.Layer1 = torch.nn.Linear(state_dim, 1000)
    self.Layer2 = torch.nn.Linear(1000, 500)
    self.Layer3 = torch.nn.Linear(500, 250)
    self.Layer4 = torch.nn.Linear(250, 100)
    self.Layer5 = torch.nn.Linear(100, 50)
    
    self.FinalLayer = torch.nn.Linear(50, action_dim)

  def forward(self, x):
    L1 = self.Layer1(x)
    ReLU1 = F.relu(L1)
    L2 = self.Layer2(ReLU1)
    ReLU2 = F.relu(L2)
    L3 = self.Layer3(ReLU2)
    ReLU3 = F.relu(L3)
    L4 = self.Layer4(ReLU3)
    ReLU4 = F.relu(L4)
    L5 = self.Layer5(ReLU4)
    ReLU5 = F.relu(L5)
    
    output = self.FinalLayer(ReLU5)
    
    return output

def GradientQLearning(env, num_episodes, Qfunction , discount_factor = 1.0,
                            epsilon = 0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""
    device = env.device
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(Qfunction.parameters()))
      
    # For every episode
    for ith_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        policy = createEpsilonGreedyPolicyGradient(Qfunction, epsilon, env.action_space.n)   
        for t in itertools.count():
            # get probabilities of all actions from current state
            action_probabilities = policy(state)
   
            # choose action according to 
            # the probability distribution
            action_index = np.random.choice(np.arange(
                      len(action_probabilities)),
                       p = action_probabilities)
            state = copy.deepcopy(state)
            # take action and get reward, transit to next state
            next_state, reward, done, SuccessF = env.step(env.actions[action_index])
            if done:
                td_target = torch.tensor([reward]).to(device)
            else:
                best_next_action = torch.argmax(Qfunction(next_state), dim = 1) 
                td_target = torch.tensor([reward]).to(device) + 0.95 * Qfunction(next_state)[:, best_next_action[0]]
                
            loss = criterion(Qfunction(state)[:, action_index], td_target.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                   
            state = next_state


            if done:
                break

    policy = createEpsilonGreedyPolicyGradient(Qfunction, 0, env.action_space.n)
       
    return Qfunction, policy

def GradientQLearningDebug(env, num_episodes, Qfunction , discount_factor = 1.0,
                            epsilon = 0.1, UpdateEpisodes = 10):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""
       
    #Qfunction = QApproxFunction(env.observation_space.n, env.action_space.n)
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = env.device
    #device = 'cpu'
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
        td_target = rewards.reshape((len(rewards), 1)) + 0.95*BestTargetValues
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