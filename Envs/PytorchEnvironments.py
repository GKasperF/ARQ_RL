#!/usr/bin/env python
# coding: utf-8

# In[1]:
import torch
import numpy as np
import gym
from gym import error, spaces, utils
import copy
from gym.utils import seeding

class iidchannel():
    def __init__(self, epsilon, batch = 1):
        self.epsilon = epsilon * torch.ones(batch)
        self.batch = batch
        self.device = 'cpu'
    
    def step(self):
        output = torch.bernoulli(1- self.epsilon)
        return output.type(torch.int64)
    def to(self, device):
        self.epsilon = self.epsilon.to(device)
        self.device = device
        return self

class GilbertElliott():
    def __init__(self, p, r, k, epsilon, batch = 1):
      self.p = p * torch.ones(batch)
      self.r = r * torch.ones(batch)
      self.epsilon = epsilon * torch.ones(batch)
      self.k = k * torch.ones(batch)
      self.state = torch.bernoulli(p * torch.ones(batch))
      self.batch = batch
      self.device = 'cpu'
      self.output = torch.zeros(batch)
    def step(self):
      indices0 = self.state == 0
      indices1 = self.state != 0
      self.output[indices0] = torch.bernoulli(1 - self.k[indices0])
      self.output[indices1] = torch.bernoulli(1 - self.epsilon[indices1])
      
      self.state[indices0] = torch.bernoulli(self.p[indices0])
      self.state[indices1] = torch.bernoulli(1 - self.r[indices1])

      return self.output.type(torch.int64)
    def reset(self):
      self.state = torch.bernoulli(self.p * torch.ones(self.batch))
      return(self.state)
    def __str__(self):
      return('Gilbert-Elliott channel with parameters {}'.format((self.p, self.r, self.epsilon, self.k)))
    def to(self, device):
      self.p = self.p.to(device)
      self.r = self.r.to(device)
      self.epsilon = self.epsilon.to(device)
      self.k = self.k.to(device)
      self.state = self.state.to(device)
      self.device = device
      self.output = self.output.to(device)
      return(self)

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
        self.channel = channel
        #action space
        self.actions = torch.tensor([1, 0])
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
        success = torch.all(torch.eq(self.agent_state, self.finish_state), dim = 1)
        reward = (torch.logical_not(success)) * (-action.reshape(len(action)) - self.alpha)

        self.agent_state[:, 0] = action.reshape((len(action),))

        temp = self.agent_state[:, self.Tf - 1].type(torch.int64).reshape(self.batch) == 1
        temp2 = self.channel.step()

        new_success = (temp2 & temp)
        success = new_success | success
        reward[new_success == 1] = reward[new_success == 1] + self.beta
        self.agent_state[success == 1] = copy.deepcopy(self.finish_state[success == 1])
        self.agent_state = torch.roll(self.agent_state, 1, 1)
        self.agent_state[ success == 0, 0] = 0
        done = torch.all(success.type(torch.uint8))
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
        self.actions = self.actions.to(device)
        self.device = device
        return(self)