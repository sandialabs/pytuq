#!/usr/bin/env python

"""Example demonstrating Rosenblatt transformation."""

import numpy as np

from pytuq.rv.rosen import Rosenblatt
import pytuq.utils.plotting as xp

# Plotting settings
xp.myrc()

#############################################
#############################################
#############################################

# Sampling model
def expu(nsam, ndim):
    return np.exp(np.random.rand(nsam, ndim))


#############################################
#############################################
#############################################

# Number of original samples used to construct Rosenblatt map
nsam = 21
# Number of samples to evaluate Rosenblatt function
neval = 99
# Number of samples for re-sampling
n_resam = 111
# Dimensionality of random variable
ndim = 2
# Sampling function
sampling_true = expu


# Original samples that are used to construct Rosenblatt map
xsam = sampling_true(nsam, ndim)
ros = Rosenblatt(xsam, sigmas=None) #sigmas=0.1*np.ones(ndim))
print(ros.sigmas)
# Plot samples
xp.plot_xrv(xsam, prefix='xsam')

####
#### Test forward Rosenblatt
####

# New samples to evaluate Rosenblatt function on
xeval = sampling_true(neval, ndim)
# New samples that are 'in-theory' uniformly distributed
xeval_unif = np.array([ros(x) for x in xeval])
#print(xeval_unif)
# Plot samples
xp.plot_xrv(xeval, prefix='xeval')
xp.plot_xrv(xeval_unif, prefix='xeval_unif')

####
#### Test inverse Rosenblatt
####

# Sample new uniform r.v.
xresam_unif = np.random.rand(n_resam,ndim)
# InvRos acts on uniform r.v. to resample from original distribution
xresam = np.array([ros.inv(u) for u in xresam_unif])
#print(xresam)
# Plot samples
xp.plot_xrv(xresam_unif, prefix='xresam_unif')
xp.plot_xrv(xresam, prefix='xresam')


# Plot PDFs to check forward Rosenblatt
xp.plot_samples_pdfs([xresam_unif, xeval_unif],
                  legends=['True Unif', 'Resam R(X)'],
                  colors=['b', 'r'],
                  file_prefix='xpdf_unif')

# Plot PDFs to check inverse Rosenblatt
xp.plot_samples_pdfs([xsam, xresam],
                  legends=['True Samples', r'Resampled R$^{-1}$(U)'],
                  colors=['b', 'r'],
                  file_prefix='xpdf')



