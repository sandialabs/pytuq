#!/usr/bin/env python


import sys
import numpy as np
import matplotlib.pyplot as plt

from pytuq.linred.kle import KLE
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


