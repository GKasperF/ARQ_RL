#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import time
import sys
import gym
from gym import error, spaces, utils
import copy
from gym.utils import seeding
import matplotlib.pyplot as plt
    
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

class EnvFeedbackGeneral(gym.Env):
    def __init__(self, Tf, alpha, beta, channel, batch):
        self.Tf = Tf
        self.alpha = alpha
        self.beta = beta
        self.channel = channel
        #action space
        self.actions = ['send', 'silence'];
        self.action_space = spaces.Discrete(2)
        self.batch = batch
        
        #observation space
        self.observation_space = spaces.MultiBinary(Tf)

        self.start_state = torch.zeros((batch, Tf))
        self.agent_state = copy.deepcopy(self.start_state)
    
        #Probability array
        self.array = {}
        
    def step(self, action):
        
        for j in range(self.batch):
        
            #Perform action at time instant t
            if action in self.actions:
              if action == 'send':
                self.agent_state[j, 0] = 1
                reward = -1 - self.alpha
              else:
                self.agent_state[j, 0] = 0
                reward = 0 - self.alpha
            else:
              print('Action not recognized')
              return()
            #Verify if there is a need to go to time instat t+1
            temp = self.agent_state[j, self.Tf-1]
            temp = temp.type(torch.int64)
            temp2 = self.channel.step()
            success = temp & temp2

            if success:
              reward = reward + self.beta
              self.agent_state = self.finish()
              return(self.agent_state, reward, 1, success)

            #Go to time instant t + 1
            for t in range(self.Tf - 1,0,-1):
              self.agent_state[j, t] = self.agent_state[j, t-1]

            self.agent_state[j, 0] = 0

        return(self.agent_state, reward, 0, success)
    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        return(self.agent_state)
    def finish(self):
        self.agent_state = 2*torch.ones((self.batch, self.Tf))
        return(self.agent_state)

class iidchannel():
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def step(self):
        output = np.random.binomial(1, 1- self.epsilon)
        return output