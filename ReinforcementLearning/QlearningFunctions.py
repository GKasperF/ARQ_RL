#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
import itertools
import plotting
import numpy as np
import copy

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
   
    stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes))
    


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
   
            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t
               
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
       
    return Q, stats, policy, NumberVisits, DebugStr

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
   
            # done is True if episode terminated   
            if done:
                break
                   
            state = next_state

    policy = createEpsilonGreedyPolicy(Q, 0, env.action_space.n)
       
    return Q, policy

