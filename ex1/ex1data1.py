#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:19:24 2017

@author: lwinchen
"""

# Importing libraries or modules 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Working directory
os.chdir('/Users/lwinchen/Desktop/Machine Learning/ML Python')

# Loading data
data = pd.read_csv('ex1data1.txt', header = None, names = ['Population', 'Profit'])

# Plotting data
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))

# Inserting ones into data
data.insert (0, 'Ones', 1)		# at 0 index, with header ‘Ones’, and adding 1 

# Assigning data to X and y
cols = data.shape[1]		# By creating cols, you can apply any data dimension
X = np.matrix(data.iloc[:,0:cols-1])
y = np.matrix(data.iloc[:,cols-1:cols])

# Initializing theta
# Making zeros for all columns except the 1st one we just added
theta = np.matrix(np.zeros([cols-1]))

# Compute Cost
def computeCost(X, y, theta):
    hypothesis = (X * theta.T)
    error = hypothesis - y
    cost = 1/(2*len(X)) * np.sum(np.power(error,2))
    return cost

computeCost(X, y, theta)

# Assigning paramters
alpha = 0.01
iters = 1500

# gradientDescent
def gradientDescent(X, y, theta, alpha, iters):
    temp_theta = np.matrix(np.zeros(theta.shape))
    num_theta = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        hypothesis = X * theta.T
        error = hypothesis - y 
        
        for j in range(num_theta):
            term = np.multiply(error, X[:,j])
            temp_theta[0,j] = temp_theta[0,j] - ((alpha/len(X)) * np.sum(term))
	  	
        theta = temp_theta    
        cost[i] = computeCost(X, y, theta)
    
    return theta, cost

g, cost = gradientDescent(X, y, theta, alpha, iters)
g

# Plotting figure
x_data = np.linspace(data.Population.min(), data.Population.max(), 100)
linearReg = g[0,0] + (g[0,1]*x_data)

fig, pp = plt.subplots(figsize=(12,8))
pp.plot(x_data, linearReg, 'r', label='Prediction')
pp.scatter(data.Population, data.Profit, label = 'Training Data')

pp.legend(loc=2)
pp.set_xlabel('Population')
pp.set_ylabel('Profit')
pp.set_title('Population Vs Predicted Profit')

# Plotting Epoch over cost
fig, ec = plt.subplots(figsize=(12,8))
ec.plot(np.arange(iters), cost, 'r')

ec.set_xlabel('Iterations')
ec.set_ylabel('Cost')
ec.set_title('Error vs Training Epoch')

# Sklearn
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y)

x = np.array(X[:, 1].A1)
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs Population by sci-kit')





