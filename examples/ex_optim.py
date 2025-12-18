#!/usr/bin/env python
"""Example demonstrating optimization algorithms on the Rosenbrock function.

This script compares different optimization methods (Gradient Descent, Adam, PSO, Scipy)
for minimizing the Rosenbrock function.
"""

import sys
import numpy as np


from pytuq.optim.gd import GD, Adam
from pytuq.optim.pso import PSO
from pytuq.optim.sciwrap import ScipyWrapper



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
method = 'Adam' #'GD', 'Adam', 'PSO', 'BFGS'


objective, objectivegrad, objectiveinfo = obj_rosenbrock, obj_rosenbrock_grad, {'a': 1.0, 'b': 100.}


if method == 'GD':
    myopt = GD(step_size=0.001)
    myopt.setObjective(objective, objectivegrad, **objectiveinfo)
    opt_results = myopt.run(nsteps, param_ini)
elif method == 'Adam':
    myopt = Adam(dim, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-2)
    myopt.setObjective(objective, objectivegrad, **objectiveinfo)
    opt_results = myopt.run(nsteps, param_ini)
elif method == 'PSO':
    myopt = PSO(dim)
    myopt.setObjective(objective, None, **objectiveinfo)
    opt_results = myopt.run(nsteps, None)
elif method == 'BFGS':
    myopt = ScipyWrapper('BFGS')
    myopt.setObjective(objective, objectivegrad, **objectiveinfo)
    opt_results = myopt.run(None, param_ini)
else:
    print("Optimization method is not recognized. Exiting.")
    sys.exit()



samples, cmode, pmode = opt_results['samples'], opt_results['best'], opt_results['bestobj']

np.savetxt('history.txt', samples)

