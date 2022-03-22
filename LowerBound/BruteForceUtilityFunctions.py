#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from LowerBound.BinaryConversion import dectobin
import copy

# In[2]:


def ProbabilityScheduling(epsilon, Tf, S):
    prob_tr = np.zeros(len(S)+1)
    prob_N = np.zeros(np.sum(S)+1)
    
    for t in range(len(S)):
        if S[t] == 1:
            prob_tr[t] = (1 - epsilon) * epsilon**(np.sum(S[range(t)]))
            last_index = np.minimum(t + Tf, len(S))
            N_temp = np.sum(S[range(last_index)])
            prob_N[N_temp] += prob_tr[t]
            
    
    prob_tr[len(S)] = 1 - np.sum(prob_tr[range(len(S))])
    
    N_temp = np.sum(S)
    prob_N[N_temp] += prob_tr[len(S)]
    
    tr = range(len(S)+1)
    Nt = range(np.sum(S)+1)
    
    Expected_Delay = np.sum( np.multiply(tr, prob_tr))
    Expected_Transmissions = np.sum( np.multiply(Nt, prob_N))
    return(Expected_Transmissions, Expected_Delay)

def ProbabilityScheduling_dec(epsilon, Tf, T, S_dec):
    
    S = dectobin(S_dec, T+1)
    
    Expected_Transmissions, Expected_Delay = ProbabilityScheduling(epsilon, Tf, S)
    return(Expected_Transmissions, Expected_Delay)


# In[3]:


def lower_convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """
    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    #return lower[:-1] + upper[:-1]
    return lower
    #return upper
    
def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return lower[:-1] + upper[:-1]
    #return lower
    #return upper

def ProbabilitySchedulingGE_Simple(alpha, beta, Tf, S):
    p0 = np.array([beta/(alpha+beta), alpha/(alpha+beta)])
    last_erasure = -1
    last_erasure_prob = 1
    P_matrix = np.array([[1 - alpha, alpha], [beta, 1 - beta]])
    
    
    prob_tr = np.zeros(len(S)+1)
    prob_N = np.zeros(np.sum(S)+1)
    
    for t in range(len(S)):
        if S[t] == 1:
            TransitionTemp = np.linalg.matrix_power(P_matrix, t - last_erasure)
            
            current_state_prob = np.matmul(p0, TransitionTemp)
            
            prob_success = current_state_prob[0]
            
            
            prob_tr[t] = prob_success * last_erasure_prob
            
            last_erasure_prob = last_erasure_prob * (1 - prob_success)
            
            p0 = np.array([0, 1])
            last_erasure = t
            
            
            last_index = np.minimum(t + Tf, len(S))
            N_temp = np.sum(S[range(last_index)])
            prob_N[N_temp] += prob_tr[t]
            
    
    prob_tr[len(S)] = 1 - np.sum(prob_tr[range(len(S))])
    
    N_temp = np.sum(S)
    prob_N[N_temp] += prob_tr[len(S)]
    
    tr = range(len(S)+1)
    Nt = range(np.sum(S)+1)
    
    Expected_Delay = np.sum( np.multiply(tr, prob_tr))
    Expected_Transmissions = np.sum( np.multiply(Nt, prob_N))
    return(Expected_Transmissions, Expected_Delay)

def ProbabilitySchedulingGE_simple_dec(alpha, beta, Tf, T, S_dec):
    
    S = dectobin(S_dec, T+1)
    
    Expected_Transmissions, Expected_Delay = ProbabilitySchedulingGE_Simple(alpha, beta, Tf, S)
    return(Expected_Transmissions, Expected_Delay)

def ProbabilitySchedulingGE_Full(alpha, beta, epsilon, h, Tf, S):
    #Assume we start each episode with a success at time t - Tf
    p0 = np.array([1, 0])
    P_matrix = np.array([[1 - alpha, alpha], [beta, 1 - beta]]) #define the transition matrix. When we have no new information, update is p0 = p0 * P_matrix
    TransitionTemp = np.linalg.matrix_power(P_matrix, -1 - (- Tf))
    p0 = np.matmul(p0, TransitionTemp) #Knowing that t - Tf was a success, assume we have no new information about erasures until time t - 1.

    last_erasure = -1
    previous_erasures_prob = 1 #Probability that previous packets have been lost
    
      
    prob_tr = np.zeros(len(S)+1)
    prob_N = np.zeros(np.sum(S)+1)
    
    for t in range(len(S)):
        if S[t] == 1:
            TransitionTemp = np.linalg.matrix_power(P_matrix, t - last_erasure) 
            Past_TransitionTemp = np.linalg.matrix_power(P_matrix, t - last_erasure-1) 

            previous_state_prob = np.matmul(p0, Past_TransitionTemp)
            
            current_state_prob = np.matmul(p0, TransitionTemp) #Compute current state belief given that no new information has been received between this transmission and the last one.

            prob_success = current_state_prob[0] * (1 - epsilon) + current_state_prob[1] * (1 - h) #Compute probability of success given current belief
            
            prob_tr[t] = prob_success * previous_erasures_prob #Compute probability of success of the current transmission *and* failure of all previous transmissions
            
            last_index = np.minimum(t + Tf, len(S))
            N_temp = np.sum(S[range(last_index)])
            prob_N[N_temp] += prob_tr[t]

            #Update for the next transmission:

            previous_erasures_prob = previous_erasures_prob * (1 - prob_success) #Compute the new probability that all transmissions fail.
            
            temp = epsilon*(1 - alpha) * previous_state_prob[0] / (1 - prob_success) + beta*epsilon * previous_state_prob[1]/(1 - prob_success) #Compute the new belief conditioning on an erasure at time t occuring
            p0 = np.array([temp, 1-temp])
            last_erasure = t
            
    
    prob_tr[len(S)] = 1 - np.sum(prob_tr[range(len(S))])
    
    N_temp = np.sum(S)
    prob_N[N_temp] += prob_tr[len(S)]
    
    tr = range(len(S)+1)
    Nt = range(np.sum(S)+1)
    
    Expected_Delay = np.sum( np.multiply(tr, prob_tr))
    Expected_Transmissions = np.sum( np.multiply(Nt, prob_N))
    return(Expected_Transmissions, Expected_Delay)

def ProbabilitySchedulingGE_full_dec(alpha, beta, epsilon, h, Tf, T, S_dec):
    #Alpha = transition from good (0) to bad (1)
    #beta = transition from bad to good
    #epsilon = erasure probability in the good state (< 0.5)
    #h = erasure probability in the bad state (>0.5)
    S = dectobin(S_dec, T+1)
    
    Expected_Transmissions, Expected_Delay = ProbabilitySchedulingGE_Full(alpha, beta, epsilon, h, Tf, S)
    return(Expected_Transmissions, Expected_Delay)

def ProbabilitySchedulingMarkovChannel(P_matrix, epsilon_vector, Tf, S):
    #Assume we start each episode with a success at time t - Tf
    p0 = np.zeros(len(epsilon_vector))
    p0[0] = 1
    TransitionTemp = np.linalg.matrix_power(P_matrix, -1 - (- Tf))
    p0 = np.matmul(p0, TransitionTemp) #Knowing that t - Tf was a success, assume we have no new information about erasures until time t - 1.

    last_erasure = -1
    previous_erasures_prob = 1 #Probability that previous packets have been lost
    prob_tr = np.zeros(len(S)+1)
    prob_N = np.zeros(np.sum(S)+1)
    
    for t in range(len(S)):
        if S[t] == 0:
            p0 = np.matmul(p0, P_matrix)
            prob_tr[t] = 0
        else:
            current_state_prob = np.matmul(p0, P_matrix) #Compute current state belief given that no new information has been received between this transmission and the last one.
            prob_success = np.dot(current_state_prob, 1- epsilon_vector)

            #prob_success = current_state_prob[0] * (1 - epsilon) + current_state_prob[1] * (1 - h) #Compute probability of success given current belief
            
            prob_tr[t] = prob_success * previous_erasures_prob #Compute probability of success of the current transmission *and* failure of all previous transmissions
            
            last_index = np.minimum(t + Tf, len(S))
            N_temp = np.sum(S[range(last_index)])
            prob_N[N_temp] += prob_tr[t]

            #Update for the next transmission:

            previous_erasures_prob = previous_erasures_prob * (1 - prob_success) #Compute the new probability that all transmissions fail.
            
            temp_probs = np.zeros(p0.shape)

            for j in range(len(epsilon_vector)):
                P_column = P_matrix[:, j]
                numerator = np.matmul(p0, P_column)
                denominator = np.matmul(np.matmul(p0, P_matrix), epsilon_vector)
                temp_probs[j] = epsilon_vector[j] * numerator/denominator

            p0 = copy.copy(temp_probs)


            #temp = epsilon*(1 - alpha) * previous_state_prob[0] / (1 - prob_success) + beta*epsilon * previous_state_prob[1]/(1 - prob_success) #Compute the new belief conditioning on an erasure at time t occuring
            #p0 = np.array([temp, 1-temp])
            
    
    prob_tr[len(S)] = 1 - np.sum(prob_tr[range(len(S))])
    
    N_temp = np.sum(S)
    prob_N[N_temp] += prob_tr[len(S)]
    
    tr = range(len(S)+1)
    Nt = range(np.sum(S)+1)
    
    Expected_Delay = np.sum( np.multiply(tr, prob_tr))
    Expected_Transmissions = np.sum( np.multiply(Nt, prob_N))
    return(Expected_Transmissions, Expected_Delay)

def ProbabilitySchedulingMarkovChannel_dec(P_matrix, epsilon_vector, Tf, T, S_dec):
    #General Markov model. P_matrix is the transition matrix of the model and epsilon_vector represents the probability of erasure for each state.
    S = dectobin(S_dec, T+1)
    
    Expected_Transmissions, Expected_Delay = ProbabilitySchedulingMarkovChannel(P_matrix, epsilon_vector, Tf, S)
    return(Expected_Transmissions, Expected_Delay)