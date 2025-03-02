#!/usr/bin/env python

"""Example demonstrating Rosenblatt transformation."""

import numpy as np
import matplotlib.pyplot as plt

from pytuq.rv.rosen import Rosenblatt
import pytuq.utils.plotting as xp

# Plotting settings
xp.myrc()

#############################################
#############################################
#############################################

# Sampling model
def expu(nsam, ndim):
    return np.exp(np.random.rand(nsam, ndim))

def standard_normal(nsam, ndim):
    return np.random.randn(nsam, ndim)

#############################################
#############################################
#############################################

# Number of original samples used to construct Rosenblatt map
nsam = 11
# Dimensionality of random variable
ndim = 1
# Sampling function
sampling_true = expu

ngr = 111


# Original samples that are used to construct Rosenblatt map
xsam = sampling_true(nsam, ndim)

ros = Rosenblatt(xsam, sigmas=None)



####
#### Plot forward Rosenblatt function
####
ygr = np.linspace(np.exp(-1.0), np.exp(1.3), ngr).reshape(-1,1)
fros_ygr = np.array([ros(x) for x in ygr])


####
#### Plot inverse Rosenblatt function
####

# Sample new uniform r.v.
xgr = np.linspace(0.00000, 1.-0.000001, ngr).reshape(-1,1)
iros_xgr = np.array([ros.inv(u) for u in xgr])
# print(ros.inv(np.array([0.0])))
# print(ros.inv(np.array([0.4])))
# print(ros.inv(np.array([0.9])))
# print(ros.inv(np.array([1.0])))


plt.plot(xgr, iros_xgr, 'b-', label='Inverse')
plt.plot(fros_ygr, ygr, 'r-', label='Forward')
plt.plot(xgr, np.exp(xgr), 'g--', label='True Map')
plt.legend()
plt.savefig('icdf.png')


