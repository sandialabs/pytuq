#!/usr/bin/env python
"""Example demonstrating multivariate normal polynomial chaos random variables.

This script creates PCRV_mvn objects with specified means and covariances,
and generates samples from the multivariate normal distribution.
"""

import numpy as np

from pytuq.rv.pcrv import PCRV_mvn

# Generate random PC coefficients for a 14-dimensional PCRV
covMatSize=14
pcfs = np.random.rand(covMatSize, covMatSize+1)

# Create PCRV_mvn from mean and covariance derived from random coefficients
pcrv_mvn = PCRV_mvn(pdim=covMatSize, mean=pcfs[:,0], cov=np.dot(pcfs[:,1:], pcfs[:,1:].T))

# Sample and print info
output_samples = pcrv_mvn.sample(20)

pcrv_mvn.printInfo()
# pdim = 14

# #cfs = np.random.rand(pdim, pdim+1)
# mypcrv = PCRV_mvn(pdim)

#mypcrv.printInfo()

# mypcrv.setRandomCfs()
# print(mypcrv.coefs)
# xx = mypcrv.sampleGerm(20)
# yy = mypcrv.evalPC(xx)
