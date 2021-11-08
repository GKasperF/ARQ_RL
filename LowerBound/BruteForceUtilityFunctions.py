#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from LowerBound.BinaryConversion import dectobin

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