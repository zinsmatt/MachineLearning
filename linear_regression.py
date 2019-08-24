#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:44:53 2019

@author: matt
"""

import numpy as np
import matplotlib.pyplot as plt



#%% Load data
data = np.loadtxt("data/cars.csv", delimiter=",", skiprows=1)

fig = plt.figure()
ax = fig.add_subplot(111)
speed = data[:, 0].reshape((-1, 1))
dist = data[:, 1].reshape((-1, 1))
plt.scatter(speed, dist)
ax.xaxis.set_label_text('Car speed')
ax.yaxis.set_label_text('Stopping distance')

min_speed = int(min(speed))
max_speed = int(max(speed)) + 1
x_range = range(min_speed - 5, max_speed + 5)

#%%
# linear regression using only the input car speed

# in a statistical reasoning, we try to find the maximum likelyhood of the input data 
# (i.e we try to find the parameters for which the likelyhood of observing the measurement is maximum)

# we suppose the noise is Gaussian and zero-centered
# maximizing the likelyhood is equivalent to minimizing the following cost function
#      sum(i from 0 to n) of (b0 + b1 * x1_i + b2 * x2_i + ... + bp * xp_i - y_i)^2
# this has a closed form solution:
#       X * Beta = y    ->       Beta = (X.T * X)^-1 * X.T * y


X = np.hstack((np.ones((speed.shape[0], 1)), speed))
y = dist

beta = np.linalg.inv(X.T @ X) @ X.T @ y

v = [beta[0] + x * beta[1] for x in x_range]


plt.plot(x_range, v, c="red")