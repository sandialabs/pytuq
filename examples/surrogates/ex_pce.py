#!/usr/bin/env python

"""This file is for testing the PCE wrapper class with scalar valued functions."""

import numpy as np
import pytuq.utils.funcbank as fcb


from pytuq.surrogates.pce import PCE
from pytuq.utils.maps import scale01ToDom
from pytuq.lreg.anl import anl
from pytuq.lreg.lreg import lsq
from pytuq.utils.mindex import get_mi


########################################################
########################################################
########################################################

N = 14                      # Number of data points to generate
order = 4                   # Polynomial order
true_model = fcb.sin4       # Function to approximate 
dim = 3                     # Dimensionality of input data
# dim = 1

########################################################
########################################################
########################################################

np.random.seed(42) 

# Domain: defining range of input variable
domain = np.ones((dim, 2))*np.array([-1.,1.])

# Generating x-y data
x = scale01ToDom(np.random.rand(N,dim), domain)
y = true_model(x)

# Testing PCE class:

# (1) Construct polynomial chaos expansion
pce = PCE(dim, order, 'LU')
# pce = PCE(dim, order, 'HG')
# pce = PCE(dim, order, ['LU', 'HG']) # dim = 2
# pce = PCE(dim, order, ['HG', 'LU', 'HG', 'LU']) # dim = 4

pce.set_training_data(x, y)

# (2) Pick method for linear regression object, defaulting to least squares regression
# print(pce.build(regression = 'anl'))
# print(pce.build(regression = 'anl', method = 'vi'))
print(pce.build())
# print(pce.build(regression = 'lsq'))
# print(pce.build(regression = 'bcs'))

# (3) Make predictions for data points and print results:
results = pce.evaluate(x)

# np.random.seed(45) # Generate a single random data point within the domain:
# single_point = scale01ToDom(np.random.rand(1, dim), domain)
# results = pce.evaluate(single_point)

# Generate random input for evaluation:
# x_eval = scale01ToDom(np.random.rand(10,dim), domain)
# results = pce.evaluate(x_eval)

print(results)