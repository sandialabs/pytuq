#!/usr/bin/env python
"""MCMC Random Variable.

Demonstrates the MCMCRV class for defining random variables via MCMC sampling.
Creates an MCMC-based random variable from a Gaussian log-posterior,
draws samples, and evaluates unscaled PDF values.
"""

import numpy as np

from pytuq.rv.mrv import MCMCRV
from pytuq.utils.plotting import plot_xrv

##################################################

dim = 2

# Gaussian log-posterior centered at (5,5)
logpost = lambda x : -0.5 * np.linalg.norm(x-5.)**2

# Build MCMC-based random variable
mcmc_rv = MCMCRV(dim, logpost, nmcmc=11111)

# Draw samples and save
xsam = mcmc_rv.sample(1000)


np.savetxt('xsam.txt', xsam)
plot_xrv(xsam)

# Evaluate unscaled PDF at drawn samples
dens = mcmc_rv.pdf_unscaled(xsam)
np.savetxt('dens.txt', dens)
