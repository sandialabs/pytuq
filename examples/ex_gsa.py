#!/usr/bin/env python
"""Example demonstrating global sensitivity analysis using Sobol indices.

This script computes main and total Sobol sensitivity indices using either
sampling-based (SamSobol) or PC-based (PCSobol) methods for a test function.
"""

import sys
import numpy as np
try:
    import pprint
except ModuleNotFoundError:
    print("Please pip install pprint for more readable printing.")


from pytuq.gsa.gsa import SamSobol, PCSobol
from pytuq.utils.plotting import plot_sens, plot_jsens, myrc

myrc()

# Setup the function of interest
myfunc = lambda x: (np.sum(x, axis=1)+np.prod(x, axis=1)).reshape(-1,1)
domain = np.tile(np.array([-1,1]), (3,1))

# myfunc = Ishigami()
# domain = myfunc.domain

dim = domain.shape[0]
# Number of samples
nsam = 1000000
# Method selection
method = "SamSobol" #"PCSobol" #"SamSobol"

# Pick a method
if method == "SamSobol":
    sMethod = SamSobol(domain)
elif method == "PCSobol":
    sMethod = PCSobol(domain, pctype='LU', order=7)
else:
    print(f"Sampling method {method} is unrecognized. Exiting.")
    sys.exit()

# Sample the input
xsam = sMethod.sample(nsam)
# Evaluate the function
ysam = myfunc(xsam)
# Compute the sensitivities
sens = sMethod.compute(ysam)

# Print the sensitivities
try:
    pprint.pprint(sens)
except NameError:
    print(sens)


# Plot main sensitivities
plot_sens(sens['main'].reshape(1,-1),range(dim),range(1))
# Plot joint sensitivities
plot_jsens(sens['main'],sens['jointt'])
