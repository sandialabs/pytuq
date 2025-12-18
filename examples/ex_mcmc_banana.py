#!/usr/bin/env python
"""Example demonstrating MCMC sampling for a banana-shaped (Rosenbrock) distribution.

This script demonstrates different MCMC methods (AMCMC, HMC, MALA) for sampling from
a challenging banana-shaped posterior distribution.
"""

import numpy as np

from scipy.stats import multivariate_normal

from pytuq.minf.mcmc import AMCMC, HMC,MALA

method='amcmc' #sys.argv[1] #'amcmc', 'hmc' or 'mala'

# Function that computes log-posterior
def logpost_mvn(x, mean=None, cov=None):
    dim = x.shape[0]
    if mean is None:
        mean = np.zeros((dim,))
    if cov is None:
        cov = np.eye(dim)
    return multivariate_normal.logpdf(x, mean=mean, cov=cov, allow_singular=False)


def logpost_rosenbrock(x, a=1.0, b=100.):

    return -(a-x[0])**2-b*(x[1]-x[0]**2)**2


def logpost_rosenbrock_grad(x, a=1.0, b=100.):

    return np.array([2*(a-x[0])+4.*b*x[0]*(x[1]-x[0]**2), -2.* b*(x[1]-x[0]**2)])




###
# Run MCMC
###


# Set the initial parameters and run MCMC
dim = 2
nmcmc = 100000  # number of MCMC samples requested
param_ini = np.random.rand(dim)  # initial parameter values

#logpost, logpostgrad, lpinfo = logpost_mvn, None, {'mean': np.ones((dim,)), 'cov': np.eye(dim)}
logpost, logpostgrad, lpinfo = logpost_rosenbrock, logpost_rosenbrock_grad, {'a': 1.0, 'b': 100.}


if method=='amcmc':
    mymcmc = AMCMC(t0=1000, tadapt=100, gamma=0.1)

elif method=='hmc':
    mymcmc = HMC(epsilon=0.05, L=3)

elif method == 'mala':
    mymcmc = MALA(epsilon=0.05)

mymcmc.setLogPost(logpost, logpostgrad, **lpinfo)
mcmc_results = mymcmc.run(nmcmc, param_ini)

samples, cmode, pmode, acc_rate = mcmc_results['chain'], mcmc_results['mapparams'], mcmc_results['maxpost'], mcmc_results['accrate']

np.savetxt('chain.txt', samples)


# # Only if 1d
# plt.plot(xcheck, ycheck_pred, 'g-', label='MAP')
# plt.plot(xcheck, ycheck, 'k--', label='Truth')
# plt.plot(xd, yd, 'ko', label='Data')
# plt.legend()
# plt.show()
