#!/usr/bin/env python
"""Example demonstrating Karhunen-Loève Expansion (KLE) and SVD for dimensionality reduction.

This script builds KLE or SVD representations of model output data to capture
variance with reduced dimensionality.
"""

import numpy as np

from pytuq.linred.svd import SVD


####################################################################
####################################################################

# Number of samples (N), dimensionality of input (d), dimensionality of output (M)
nsam, ndim, nout = 111, 5, 33
# Input parameter
ptrain = 2.*np.random.rand(nsam, ndim)-1
# Toy model: f_i = \exp(\sum_{j=1}^d a_{ij} p_j) for some random a_{ij}, for i=1,..., M.
ytrain = np.exp(np.dot(ptrain, np.random.rand(ndim, nout)))


## Build KLE
#kl = KLE()

## or, Build SVD
kl = SVD()

kl.build(ytrain.T, plot=True)
## Get the number of eigenvalues that capture 99% of variance
neig = kl.get_neig(0.99)
## Evaluate the truncated KL up to the selected number of eigenvalues
ytrain_kl = kl.eval(neig=neig)
## Plot explained variance
kl.plot_expvar()


