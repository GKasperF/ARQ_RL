#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def dectobin(dec, n):
    if(np.mod(dec, 1) != 0):
        print('Error: input is not integer')
        return()
    if(dec >=  2**n):
        print('Error: input can not be represented')
        return()
    if(dec < 0):
        print('Error: input can not be represented (only positive numbers)')
        return()
    
    binaryarray = np.zeros((n, 1))
    binaryarray = binaryarray.astype(int)

    for i in range(n):
        binaryarray[i,0] = np.mod(dec, 2)
        dec = (dec - binaryarray[i,0])/2

    return(binaryarray.copy())

def bintodec(binvalue):
    dec = np.zeros(1).astype(int)
    for i in range( binvalue.size):
        dec += binvalue[i, 0] * (2**i)
        
    return(dec)

