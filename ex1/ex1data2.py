#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:25:58 2017

@author: lwinchen
"""

# Importing modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Working directory
os.chdir('/Users/lwinchen/Desktop/Machine Learning/ML Python')

# Loading data
data = pd.read_csv('ex1data2.txt', header=None, names=['Size', 'Bedrooms', 'Price'])

# Assign each feature
cols = data.shape[1]
Size = data.iloc[:,0]
Bedrooms = data.iloc[:,1]
Price = data.iloc[:,2]

# Viewing relevance of each data against price
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(Size, Price)
ax2.scatter(Bedrooms, Price)

# Normalize features, X
cols = data.shape[1]
X_raw = data.iloc[:,0:cols-1]
X_norm = (X_raw - X_raw.mean()) / X_raw.std()

# Adding ones to the data
X_norm.insert (0,'Ones',1)

# Assign data 
X = np.matrix(X_norm.values)
y = np.matrix(data.iloc[:,cols-1:cols])
theta = np.matrix(np.zeros([cols]))

# Compute Cost
def computeCost(X, y, theta):
    hypothesis = X * theta.T
    error = hypothesis - y
    cost = 1/(2*len(X)) * np.sum(np.power(error,2))
    return cost

cost_0 = computeCost(X, y, theta)
print ('Error cost at theta zeros =', cost_0)

# Parameters for gradient descent
alpha = 0.01
iters = 1500

# Gradient descent
def gradientDescent(X, y, theta, alpha, iters):
    temp_theta = np.matrix(np.zeros(theta.shape[1]))
    num_theta = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        hypothesis = X * theta.T
        error = hypothesis - y
        
        for j in range(num_theta):
            term = np.multiply(error,X[:,j])
            temp_theta [0,j] = temp_theta[0,j] - ((alpha/len(X)) * np.sum(term))
            
        theta = temp_theta
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

g, cost = gradientDescent(X, y, theta, alpha, iters)
print('Gradient at minimal error cost', g)
print('Minimal cost at optimal gradient', cost)

# Drawing Iterations vs Error
fig, ie = plt.subplots(figsize=(12,8))
ie.plot(np.arange(iters),cost, 'r')
ie.set_xlabel('Iterations')
ie.set_ylabel('Cost')
ie.set_title('Error vs Training Epoch')








