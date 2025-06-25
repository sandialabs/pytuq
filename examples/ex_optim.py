#!/usr/bin/env python

import numpy as np


from pytuq.optim.gd import GD


# Function that computes objective
def obj_rosenbrock(x, a=1.0, b=100.):

    return (a-x[0])**2+b*(x[1]-x[0]**2)**2

# Function that computes objective gradient
def obj_rosenbrock_grad(x, a=1.0, b=100.):

    return -np.array([2*(a-x[0])+4.*b*x[0]*(x[1]-x[0]**2), -2.* b*(x[1]-x[0]**2)])




###
# Run Optimization
###


# Set the initial parameters and run Optimization
dim = 2
nsteps = 10000  # number of steps
param_ini = np.random.rand(dim)  # initial parameter values

#logpost, logpostgrad, lpinfo = logpost_mvn, None, {'mean': np.ones((dim,)), 'cov': np.eye(dim)}
objective, objectivegrad, objectiveinfo = obj_rosenbrock, obj_rosenbrock_grad, {'a': 1.0, 'b': 100.}


myopt = GD(step_size=0.001)

myopt.setObjective(objective, objectivegrad, **objectiveinfo)
opt_results = myopt.run(nsteps, param_ini)

samples, cmode, pmode = opt_results['samples'], opt_results['best'], opt_results['bestobj']

np.savetxt('history.txt', samples)

