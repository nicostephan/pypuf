# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:14:22 2017

@author: Nico Stephan
"""

from scipy.optimize import linprog
from numpy import array, shape, zeros, sqrt

def findCenter(challenges, responses):
    for i in range(len(responses)):
        if(responses[i]>0):
            challenges[i]=-1*challenges[i]
    numberOfBits=challenges[0].shape[0]       
   
    A_ub_dim=(len(challenges),numberOfBits+1)
    A_ub=zeros(A_ub_dim)
    for i in range(A_ub_dim[0]):
        for j in range(A_ub_dim[1]):
            if(j<numberOfBits):
                A_ub[i][j]=challenges[i][j]
            else:
                A_ub[i][j]=1./sqrt(numberOfBits)
    
#    print(A_ub)
    
    
    targetFunction=zeros(numberOfBits+1)
    targetFunction[numberOfBits]=-1
    
    b_ub=zeros(len(challenges))
    
    
    bounds=[]
    for i in range (numberOfBits):
        bounds.append((-1,1))
    bounds.append((None,None))
    
    result=linprog(targetFunction,A_ub=A_ub,b_ub=b_ub,bounds=bounds)
#    print(result.x)
#    print(result.status)
    ret=zeros(result.x.size-1)
    for i in range(ret.size):
        ret[i]=result.x[i]
#    print(ret)
    return (ret,result.x[result.x.size-1])
