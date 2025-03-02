#!/usr/bin/env python


import os, sys
import numpy as np
try:
    import pprint
except ModuleNotFoundError:
    print("Please pip install pprint for more readable printing.")


from pytuq.func.benchmark import Ishigami
from pytuq.gsa.gsa import SamSobol, PCSobol, model_sens
from pytuq.utils.plotting import plot_sens, plot_jsens, myrc

myrc()

# really a trivial multioutput model, y_i = a*x_i+b, with the same number of outputs as input dimensionality, so S_{ij}=\delta_{ij} # kronecker delta.
def model(x, model_params):
    a, b = model_params
    return a*x+b

domain = np.tile(np.array([-1,1]), (3,1))
sens_main, sens_tot = model_sens(model, [3.0, -1.0], domain, method='SamSobol', nsam=100000, plot=True)
