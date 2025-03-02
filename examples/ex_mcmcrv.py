#!/usr/bin/env python
"""An example for sampling with MCMC."""

import numpy as np

from pytuq.rv.mrv import MCMCRV
from pytuq.utils.plotting import plot_xrv

##################################################

dim = 2

logpost = lambda x : -0.5 * np.linalg.norm(x-5.)**2

mcmc_rv = MCMCRV(dim, logpost, nmcmc=11111)

xsam = mcmc_rv.sample(1000)


np.savetxt('xsam.txt', xsam)
plot_xrv(xsam)

dens = mcmc_rv.pdf_unscaled(xsam)
np.savetxt('dens.txt', dens)
