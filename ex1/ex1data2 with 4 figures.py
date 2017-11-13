#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:32:42 2017

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
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = 'col', sharey= 'row')
ax1.scatter(Size, Price)
ax3.plot(Size,Price)
ax2.scatter(Bedrooms, Price)
ax4.plot(Bedrooms, Price)
ax1.set_title('Each feature vs Price')