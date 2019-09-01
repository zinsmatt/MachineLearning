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


plt.plot(x_range, v, c="red", label="linear regression")

#%%
# linear regression with augmented inputs
# the input is [1 x x²]
# this is still a linear regression wrt. the input

X = np.hstack((np.ones((speed.shape[0], 1)), speed, speed**2))
y = dist;
beta = np.linalg.inv(X.T @ X) @ X.T @ y

v = [beta[0] + x * beta[1] + x**2 * beta[2] for x in x_range]

plt.plot(x_range, v, c="green", label="augmented inputs")

plt.legend()

#%%
# linear resgression with radial basis functions aroung each input point
# we set some radial basis functions centered around some of the input points

sigma = 2.0
RBF_centers = speed[[1, 10, 15, 20, 35, 45, 49]]

def RadialBasisFunction(x):
    global sigma, RBF_centers
    return np.exp(-(np.tile(x, (RBF_centers.shape[0], 1)) - RBF_centers)**2 / (2*sigma**2))


rbf = [RadialBasisFunction(x) for x in speed]
rbf = np.hstack(rbf).T

X = np.hstack((np.ones((speed.shape[0], 1)), rbf))
y = dist

beta = np.linalg.inv(X.T @ X) @ X.T @ y

v = [beta.T.dot(np.vstack((np.ones((1, 1)), RadialBasisFunction(x))))[0] for x in x_range]

plt.plot(x_range, v, c="orange", label="radial basis functions")


# Tikhonov regularization
regularizationFactor = 8
reg = regularizationFactor * np.eye(X.shape[1], dtype=np.float)
beta_reg = np.linalg.inv(X.T @ X + reg) @ X.T @ y
v_reg = [beta_reg.T.dot(np.vstack((np.ones((1, 1)), RadialBasisFunction(x))))[0] for x in x_range]

plt.plot(x_range, v_reg, linestyle='dashed', c="orange", label="Tikhonov regularized RBF")

plt.legend()