#!/usr/bin/env python
"""Example demonstrating domain-constrained sampling from Gaussian Mixture Models.

This script samples from a GMM within a specified domain and visualizes
the resulting samples and probability densities.
"""

import numpy as np

from pytuq.rv.mrv import GMM
from pytuq.utils.plotting import plot_xrv
dim = 2
means = [np.array([5.,2.]), np.array([2.,7.]), np.array([-4.,2.])]

gmm_rv = GMM(means, weights=np.array([3., 1., 5.]) )


domain = np.array([[-4.,6], [1.,8.]])
nsam = 11111
xsam = gmm_rv.sample_indomain(nsam, domain=domain)

np.savetxt('xsam.txt', xsam)
plot_xrv(xsam)

dens = gmm_rv.pdf(xsam)
np.savetxt('dens.txt', dens)
