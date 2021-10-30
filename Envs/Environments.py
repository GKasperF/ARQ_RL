#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import sys
import gym
from gym import error, spaces, utils
import copy
from gym.utils import seeding
import matplotlib.pyplot as plt


# In[2]:


class GridEnv(gym.Env):
    def __init__(self, T, Tf, epsilon, alpha, beta):

        self.T = T
        self.Tf = Tf
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        
        #action space
        self.actions = ['send', 'silence'];
        self.action_space = spaces.Discrete(2)
        
        
        #observation space
        self.observation_space = spaces.Tuple((spaces.MultiBinary(T+1), spaces.Discrete(T+1)))

        self.start_state = np.zeros((T+2))
        self.start_state = self.start_state.astype(int)
        self.agent_state = copy.deepcopy(self.start_state)
    
        #Probability array
        self.array = {}
                
    def step(self, action):
        if action in self.actions:
          if action == 'send':
            self.agent_state[self.agent_state[self.T+1]] = 1
            reward = -1
          else:
            reward = 0
        else:
          print('Action not recognized')
          return()
        
        if self.agent_state[self.T+1]+1 - self.Tf >= 0:
          temp = self.agent_state[self.agent_state[self.T+1]+1 - self.Tf]
          temp2 = np.random.binomial(1, 1- self.epsilon)
          success = temp & temp2
        else:
          success = 0

        if success:
          reward = reward + self.beta * self.alpha**(self.agent_state[self.T+1]+1 - self.Tf)
          self.agent_state = self.finish()
          return(self.agent_state, reward, 1, success)
        elif self.agent_state[self.T+1] >= self.T :
          for t in range(self.T + 1, self.T + self.Tf+1):
            temp = self.agent_state[t - self.Tf]
            temp2 = np.random.binomial(1, 1- self.epsilon)
            success = temp & temp2
            if success:
              reward = reward + self.beta * self.alpha**(t - self.Tf)
              self.agent_state = self.finish()
              return(self.agent_state, reward, 1, success)
          self.agent_state = self.finish()
          return(self.agent_state, reward, 1, success)




        self.agent_state[self.T+1] = self.agent_state[self.T+1] + 1
        return(self.agent_state, reward, 0, success)
    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        return(self.agent_state)
    def finish(self):
        self.agent_state = np.ones((self.T+2))
        return(self.agent_state)


# In[3]:


class GridEnvLin(gym.Env):
    def __init__(self, T, Tf, epsilon, alpha, beta):

        self.T = T
        self.Tf = Tf
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        
        #action space
        self.actions = ['send', 'silence'];
        self.action_space = spaces.Discrete(2)
        
        
        #observation space
        self.observation_space = spaces.Tuple((spaces.MultiBinary(T+1), spaces.Discrete(T+1)))

        self.start_state = np.zeros((T+2))
        self.start_state = self.start_state.astype(int)
        self.agent_state = copy.deepcopy(self.start_state)
    
        #Probability array
        self.array = {}
                
    def step(self, action):
        if action in self.actions:
          if action == 'send':
            self.agent_state[self.agent_state[self.T+1]] = 1
            reward = -1 - self.alpha
          else:
            reward = 0 - self.alpha
        else:
          print('Action not recognized')
          return()
        
        if self.agent_state[self.T+1]+1 - self.Tf >= 0:
          temp = self.agent_state[self.agent_state[self.T+1]+1 - self.Tf]
          temp2 = np.random.binomial(1, 1- self.epsilon)
          success = temp & temp2
        else:
          success = 0

        if success:
          reward = reward + self.beta
          self.agent_state = self.finish()
          return(self.agent_state, reward, 1, success)
        elif self.agent_state[self.T+1] >= self.T :
          for t in range(self.T + 1, self.T + self.Tf+1):
            temp = self.agent_state[t - self.Tf]
            temp2 = np.random.binomial(1, 1- self.epsilon)
            success = temp & temp2
            if success:
              reward = reward + self.beta
              self.agent_state = self.finish()
              return(self.agent_state, reward, 1, success)
            reward = reward - self.alpha
          self.agent_state = self.finish()
          return(self.agent_state, reward, 1, success)




        self.agent_state[self.T+1] = self.agent_state[self.T+1] + 1
        return(self.agent_state, reward, 0, success)
    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        return(self.agent_state)
    def finish(self):
        self.agent_state = np.ones((self.T+2))
        return(self.agent_state)


# In[4]:


class GridEnvFeedback(gym.Env):
    def __init__(self, Tf, epsilon, alpha, beta):

        self.Tf = Tf
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        
        #action space
        self.actions = ['send', 'silence'];
        self.action_space = spaces.Discrete(2)
        
        
        #observation space
        self.observation_space = spaces.MultiBinary(Tf)

        self.start_state = np.zeros((Tf))
        self.start_state = self.start_state.astype(int)
        self.agent_state = copy.deepcopy(self.start_state)
    
        #Probability array
        self.array = {}
                
    def step(self, action):
        #Perform action at time instant t
        if action in self.actions:
          if action == 'send':
            self.agent_state[0] = 1
            reward = -1 - self.alpha
          else:
            self.agent_state[0] = 0
            reward = 0 - self.alpha
        else:
          print('Action not recognized')
          return()
        #Verify if there is a need to go to time instat t+1
        temp = self.agent_state[self.Tf-1]
        temp2 = np.random.binomial(1, 1- self.epsilon)
        success = temp & temp2

        if success:
          reward = reward + self.beta
          self.agent_state = self.finish()
          return(self.agent_state, reward, 1, success)

        #Go to time instant t + 1
        for t in range(self.Tf - 1,0,-1):
          self.agent_state[t] = self.agent_state[t-1]

        self.agent_state[0] = 0
        
        
        

        

        return(self.agent_state, reward, 0, success)
    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        return(self.agent_state)
    def finish(self):
        self.agent_state = 2*np.ones((self.Tf))
        self.agent_state = self.agent_state.astype(int)
        return(self.agent_state)


# In[5]:


class GridEnvFeedbackGE(gym.Env):
    def __init__(self, Tf, alpha, beta, p, r, k, epsilon):

        self.Tf = Tf
        self.alpha = alpha
        self.beta = beta
        
        self.GE = GilbertElliott(p, r, k, epsilon)
        
        #action space
        self.actions = ['send', 'silence'];
        self.action_space = spaces.Discrete(2)
        
        
        #observation space
        self.observation_space = spaces.MultiBinary(Tf)

        self.start_state = np.zeros((Tf))
        self.start_state = self.start_state.astype(int)
        self.agent_state = copy.deepcopy(self.start_state)
    
        #Probability array
        self.array = {}
                
    def step(self, action):
        #Perform action at time instant t
        if action in self.actions:
          if action == 'send':
            self.agent_state[0] = 1
            reward = -1 - self.alpha
          else:
            self.agent_state[0] = 0
            reward = 0 - self.alpha
        else:
          print('Action not recognized')
          return()
        #Verify if there is a need to go to time instat t+1
        temp = self.agent_state[self.Tf-1]
        temp2 = self.GE.step()
        success = temp & temp2

        if success:
          reward = reward + self.beta
          self.agent_state = self.finish()
          return(self.agent_state, reward, 1, success)

        #Go to time instant t + 1
        for t in range(self.Tf - 1,0,-1):
          self.agent_state[t] = self.agent_state[t-1]

        self.agent_state[0] = 0

        return(self.agent_state, reward, 0, success)
    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        self.GE.reset()
        return(self.agent_state)
    def finish(self):
        self.agent_state = 2*np.ones((self.Tf))
        self.agent_state = self.agent_state.astype(int)
        self.GE.reset()
        return(self.agent_state)
    
class GilbertElliott():
    def __init__(self, p, r, k, epsilon):
      self.p = p
      self.r = r
      self.epsilon = epsilon
      self.k = k
      self.state = np.random.binomial(1, self.p)

    def step(self):
      if self.state == 0:
        self.state = np.random.binomial(1, self.p)
        output = np.random.binomial(1, 1-self.k)
      else:
        self.state = np.random.binomial(1, 1-self.r)
        output = np.random.binomial(1, 1-self.epsilon)
      
      return(output)
    def reset(self):
      self.state = np.random.binomial(1, self.p)
      return(self.state)


# In[6]:


class Fritchman():
    def __init__(self, alpha, beta, epsilon, M):
        self.state = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.M = M
        
    def step(self):
        if self.state == 0:
            transition = np.random.binomial(1, self.alpha)
            self.state = self.state + transition
            output = np.random.binomial(1, 1 - self.epsilon)
        else:
            transition = np.random.binomial(1, self.beta)
            output = np.random.binomial(1, self.epsilon)
            if self.state == self.M:
                if transition:
                    self.state = 0
            else:
                self.state = self.state + transition
                
        return(output)
    
class GridEnvFeedbackFritchman(gym.Env):
    def __init__(self, Tf, alpha, beta, alpha_fritchman, beta_fritchman, epsilon, M):

        self.Tf = Tf
        self.alpha = alpha
        self.beta = beta
        
        self.FM = Fritchman(alpha_fritchman, beta_fritchman, epsilon, M)
        
        #action space
        self.actions = ['send', 'silence'];
        self.action_space = spaces.Discrete(2)
        
        #observation space
        self.observation_space = spaces.MultiBinary(Tf)

        self.start_state = np.zeros((Tf))
        self.start_state = self.start_state.astype(int)
        self.agent_state = copy.deepcopy(self.start_state)
    
        #Probability array
        self.array = {}
                
    def step(self, action):
        #Perform action at time instant t
        if action in self.actions:
          if action == 'send':
            self.agent_state[0] = 1
            reward = -1 - self.alpha
          else:
            self.agent_state[0] = 0
            reward = 0 - self.alpha
        else:
          print('Action not recognized')
          return()
        #Verify if there is a need to go to time instat t+1
        temp = self.agent_state[self.Tf-1]
        temp2 = self.FM.step()
        success = temp & temp2

        if success:
          reward = reward + self.beta
          self.agent_state = self.finish()
          return(self.agent_state, reward, 1, success)

        #Go to time instant t + 1
        for t in range(self.Tf - 1,0,-1):
          self.agent_state[t] = self.agent_state[t-1]

        self.agent_state[0] = 0

        return(self.agent_state, reward, 0, success)
    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        return(self.agent_state)
    def finish(self):
        self.agent_state = 2*np.ones((self.Tf))
        self.agent_state = self.agent_state.astype(int)
        return(self.agent_state)


# In[7]:


class EnvFeedbackGeneral(gym.Env):
    def __init__(self, Tf, alpha, beta, channel):
        self.Tf = Tf
        self.alpha = alpha
        self.beta = beta
        self.channel = channel
        #action space
        self.actions = ['send', 'silence'];
        self.action_space = spaces.Discrete(2)
        
        #observation space
        self.observation_space = spaces.MultiBinary(Tf)

        self.start_state = np.zeros((Tf))
        self.start_state = self.start_state.astype(int)
        self.agent_state = copy.deepcopy(self.start_state)
    
        #Probability array
        self.array = {}
        
    def step(self, action):
        #Perform action at time instant t
        if action in self.actions:
          if action == 'send':
            self.agent_state[0] = 1
            reward = -1 - self.alpha
          else:
            self.agent_state[0] = 0
            reward = 0 - self.alpha
        else:
          print('Action not recognized')
          return()
        #Verify if there is a need to go to time instat t+1
        temp = self.agent_state[self.Tf-1]
        temp2 = self.channel.step()
        success = temp & temp2

        if success:
          reward = reward + self.beta
          self.agent_state = self.finish()
          return(self.agent_state, reward, 1, success)

        #Go to time instant t + 1
        for t in range(self.Tf - 1,0,-1):
          self.agent_state[t] = self.agent_state[t-1]

        self.agent_state[0] = 0

        return(self.agent_state, reward, 0, success)
    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        return(self.agent_state)
    def finish(self):
        self.agent_state = 2*np.ones((self.Tf))
        self.agent_state = self.agent_state.astype(int)
        return(self.agent_state)
class iidchannel():
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def step(self):
        output = np.random.binomial(1, 1- self.epsilon)
        return output
