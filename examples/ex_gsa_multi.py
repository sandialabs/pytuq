#!/usr/bin/env python
"""Example demonstrating global sensitivity analysis for multi-output models.

This script performs Sobol sensitivity analysis on a simple multi-output model
to compute main and total sensitivity indices for each output dimension.
"""

import numpy as np


from pytuq.gsa.gsa import model_sens
from pytuq.utils.plotting import myrc

myrc()

# really a trivial multioutput model, y_i = a*x_i+b, with the same number of outputs as input dimensionality, so S_{ij}=\delta_{ij} # kronecker delta.
def model(x, model_params):
    a, b = model_params
    return a*x+b

domain = np.tile(np.array([-1,1]), (3,1))
sens_main, sens_tot = model_sens(model, [3.0, -1.0], domain, method='SamSobol', nsam=100000, plot=True)
