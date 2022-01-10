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
    def __init__(self, alpha, beta, epsilon, M, batch = 1):
        self.state = torch.zeros(batch)
        self.alpha = alpha * torch.ones(batch)
        self.beta = beta * torch.ones(batch)
        self.epsilon = epsilon * torch.ones(batch)
        self.M = torch.tensor(M)
        self.device = 'cpu'
        self.batch = batch
        self.output = torch.zeros(batch)

    def step(self):
        indices = []
        for i in range(self.M):
            indices.append( self.state == i)
        
        self.output[indices[0]] = torch.bernoulli(1 - self.epsilon)[indices[0]]
        self.state[indices[0]] = self.state[indices[0]] + torch.bernoulli(self.alpha)[indices[0]]
        for i in range(1, self.M):
            self.output[indices[i]] = torch.bernoulli(self.epsilon)[indices[i]]
            self.state[indices[i]] = torch.fmod(self.state[indices[i]] + torch.bernoulli(self.beta)[indices[i]], self.M)

        return self.output.type(torch.int64)

    def reset(self):
      self.state = torch.zeros(self.batch).to(self.device)
      return(self.state)
    def __str__(self):
      return('Fritchman channel with parameters {} at device {}'.format((self.alpha, self.beta, self.epsilon, self.M), self.device))
    def to(self, device):
      self.device = device
      self.state = self.state.to(device)
      self.alpha = self.alpha.to(device)
      self.beta = self.beta.to(device)
      self.epsilon = self.epsilon.to(device)
      self.output = self.output.to(device)
      self.M = self.M.to(device)
      return(self)

class EnvFeedbackGeneral(gym.Env):
    def __init__(self, Tf, alpha, beta, channel, batch, M = 0):
        self.Tf = torch.tensor([Tf])
        self.alpha = torch.tensor([alpha])
        self.beta = torch.tensor([beta])
        self.channel = channel
        #action space
        self.actions = torch.tensor([1, 0])
        self.action_space = spaces.Discrete(2)
        self.batch = batch
        self.device = 'cpu'
        self.M = M
        
        #observation space
        self.observation_space = spaces.MultiBinary(Tf+M)

        self.start_state = torch.zeros((batch, Tf+M))
        self.agent_state = copy.deepcopy(self.start_state)
        self.finish_state = 2*torch.ones((batch, Tf+M))
    
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
    def reset_success(self):
        success_indices = torch.all(torch.eq(self.agent_state, self.finish_state), dim = 1)
        self.agent_state[success_indices] = copy.deepcopy(self.start_state[success_indices])
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

class EnvFeedbackCheating_GE(gym.Env):
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
        self.observation_space = spaces.MultiBinary(2*Tf)
        self.TransitionMatrix = torch.tensor([[ 1 - channel.p[0], channel.p[0]], [channel.r[0], 1 - channel.r[0]]])
        self.p = 0.5*torch.ones((batch, 2))
        self.start_state = torch.zeros((batch, 2*Tf))
        self.start_state[:, Tf:2*Tf] = 0.5
        self.agent_state = copy.deepcopy(self.start_state)
        self.finish_state = 2*torch.ones((batch, 2*Tf))
        self.failure_state = torch.tensor([[0.0, 1.0]]).repeat(batch, 1)
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

        observed_failure = (temp & (temp2 == 0))
        self.p[observed_failure == 1] = torch.matmul(self.failure_state[observed_failure==1], self.TransitionMatrix)
        self.p[observed_failure == 0] = torch.matmul(self.p[observed_failure == 0], self.TransitionMatrix)

        p_temp = copy.deepcopy(self.p)
        for i in range( self.agent_state.size(1)-1, self.Tf-1, -1):
          self.agent_state[success==0, i] = p_temp[success==0, 0]
          p_temp = torch.matmul(p_temp, self.TransitionMatrix)

        done = torch.all(success.type(torch.uint8))
        return(self.agent_state, reward, done, success)
    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        return(self.agent_state)
    def reset_success(self):
        success_indices = torch.all(torch.eq(self.agent_state, self.finish_state), dim = 1)
        self.agent_state[success_indices] = copy.deepcopy(self.start_state[success_indices])
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
        self.TransitionMatrix = self.TransitionMatrix.to(device)
        self.p = self.p.to(device)
        self.failure_state = self.failure_state.to(device)
        return(self)

class EnvFeedbackCheating_Noisy_GE(gym.Env):
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
        self.observation_space = spaces.MultiBinary(2*Tf)
        self.TransitionMatrix = torch.tensor([[ 1 - channel.p[0], channel.p[0]], [channel.r[0], 1 - channel.r[0]]])
        self.p = 0.5*torch.ones((batch, 2))
        self.start_state = torch.zeros((batch, 2*Tf))
        self.start_state[:, Tf:2*Tf] = 0.5
        self.agent_state = copy.deepcopy(self.start_state)
        self.finish_state = 2*torch.ones((batch, 2*Tf))
        self.failure_state = torch.tensor([[0.0, 1.0]]).repeat(batch, 1)
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

        observed_failure = (temp & (temp2 == 0))
        self.p[observed_failure == 1] = torch.matmul(self.failure_state[observed_failure==1], self.TransitionMatrix)
        self.p[observed_failure == 0] = torch.matmul(self.p[observed_failure == 0], self.TransitionMatrix)

        p_temp = copy.deepcopy(self.p)
        for i in range( self.agent_state.size(1)-1, self.Tf-1, -1):
          self.agent_state[success==0, i] = p_temp[success==0, 0] + torch.rand( sum(success==0)).to(self.device)/50
          p_temp = torch.matmul(p_temp, self.TransitionMatrix)

        done = torch.all(success.type(torch.uint8))
        return(self.agent_state, reward, done, success)
    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        return(self.agent_state)
    def reset_success(self):
        success_indices = torch.all(torch.eq(self.agent_state, self.finish_state), dim = 1)
        self.agent_state[success_indices] = copy.deepcopy(self.start_state[success_indices])
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
        self.TransitionMatrix = self.TransitionMatrix.to(device)
        self.p = self.p.to(device)
        self.failure_state = self.failure_state.to(device)
        return(self)

class EnvFeedbackRNN_GE(gym.Env):
    def __init__(self, Tf, alpha, beta, channel, ChannelModel, batch):
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
        self.observation_space = spaces.MultiBinary(2*Tf)
        
        self.h = torch.zeros((ChannelModel.num_layers, batch, ChannelModel.hidden_size))
        self.c = torch.zeros((ChannelModel.num_layers, batch, ChannelModel.hidden_size))
        self.ChannelModel = ChannelModel

        self.start_state = torch.zeros((batch, 2*Tf))
        self.start_state[:, Tf:2*Tf] = 0.5
        self.agent_state = copy.deepcopy(self.start_state)
        self.finish_state = 2*torch.ones((batch, 2*Tf))
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


        temp2 = temp.reshape(self.batch, 1).type(torch.float)
        temp = new_success.reshape(self.batch, 1).type(torch.float)

        state_in = torch.cat((temp, temp2), dim=1).reshape(self.batch, 1, 2)

        h_in = self.h
        c_in = self.c

        estimate, (h_out, c_out) = self.ChannelModel(state_in, h_in, c_in)

        self.h = h_out
        self.c = c_out
        estimate = estimate.reshape(self.batch, self.Tf)
        self.agent_state[success == 0, self.Tf:] = estimate[success == 0, :]

        done = torch.all(success.type(torch.uint8))
        return(self.agent_state, reward, done, success)
    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        return(self.agent_state)
    def reset_success(self):
        success_indices = torch.all(torch.eq(self.agent_state, self.finish_state), dim = 1)
        self.agent_state[success_indices] = copy.deepcopy(self.start_state[success_indices])
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
        self.h = self.h.to(device)
        self.c = self.c.to(device)
        self.ChannelModel = self.ChannelModel.to(device)
        return(self)