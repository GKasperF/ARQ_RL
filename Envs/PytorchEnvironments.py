#!/usr/bin/env python
# coding: utf-8

# In[1]:
import torch
import numpy as np
import gym
from gym import error, spaces, utils
import copy
from gym.utils import seeding
    
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
    def __str__(self):
      return('Gilbert-Elliott channel with parameters {}'.format((self.p, self.r, self.epsilon, self.k)))

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
        self.Tf = torch.tensor([Tf])
        self.alpha = torch.tensor([alpha])
        self.beta = torch.tensor([beta])
        self.channel = []
        for i in range(batch):
          self.channel.append(copy.deepcopy(channel))
        #action space
        self.actions = ['send', 'silence'];
        self.action_space = spaces.Discrete(2)
        self.batch = batch
        self.device = 'cpu'
        
        #observation space
        self.observation_space = spaces.MultiBinary(Tf)

        self.start_state = torch.zeros((batch, Tf))
        self.agent_state = copy.deepcopy(self.start_state)
        self.finish_state = 2*torch.ones((batch, Tf))
    
        #Probability array
        self.array = {}
        
    def step(self, action):
        reward = torch.zeros((self.batch, 1)).to(self.device)
        success = torch.zeros((self.batch, 1)).type(torch.int64).to(self.device)
        if not isinstance(action, list):
          raise TypeError('Expected a list of actions, instead got type {}'.format(type(action)))
        if len(action) != self.batch:
          raise TypeError('Expected a list of length {}, instead, got length {}'.format(self.batch, len(action)))

        for j in range(self.batch):
            #Perform action at time instant t
            if torch.all( torch.eq(  self.agent_state[j], self.finish_state[j] ) ):
              success[j] = True
              continue 

            if action[j] in self.actions:
              if action[j] == 'send':
                self.agent_state[j, 0] = 1
                reward[j] = -1 - self.alpha
              else:
                self.agent_state[j, 0] = 0
                reward[j] = 0 - self.alpha
            else:
              raise ValueError('Action should be either \'send\' or \'silence\'. Got {}'.format(print(action)))
            #Verify if there is a need to go to time instat t+1
            temp = self.agent_state[j, self.Tf-1]
            temp = temp.type(torch.int64)
            channel = self.channel[j]
            temp2 = channel.step()
            success[j] = temp & temp2

            if success[j]:
              reward[j] = reward[j] + self.beta
              self.agent_state[j] = self.finish_state[j]
            else: 
              #Go to time instant t + 1
              for t in range(self.Tf - 1,0,-1):
                self.agent_state[j, t] = self.agent_state[j, t-1]
              self.agent_state[j, 0] = 0
        done = torch.tensor([1]).to(self.device)
        for j in range(self.batch):
          done = done & success[j]

        return(self.agent_state, reward, done, success)
    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        return(self.agent_state)
    def finish(self):
        self.agent_state = copy.deepcopy(self.finish_state)
        return(self.agent_state)
    def to(self, device):
        self.start_state = self.start_state.to(device)
        self.agent_state = self.agent_state.to(device)
        self.finish_state = self.finish_state.to(device)
        self.Tf = self.Tf.to(device)
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        self.device = device
        return(self)

class iidchannel():
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def step(self):
        output = np.random.binomial(1, 1- self.epsilon)
        return output