#!/usr/bin/env python
"""Example demonstrating MCMC-based Bayesian linear model calibration.

This script uses Adaptive MCMC to calibrate a linear model to noisy data.
"""

import sys
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import minimize

from pytuq.minf.mcmc import AMCMC
from pytuq.utils.plotting import lighten_color


# Logarithm of normal PDF
def log_norm_pdf(x, mu, sigma):
    s2 = sigma * sigma
    x_mu = x - mu
    norm_const = -0.5 * np.log(2 * np.pi * s2)
    return (norm_const - 0.5 * x_mu * x_mu / s2)



# Model that is being calibrated
def linear_model(x, par):
    a = par[0]
    b = par[1]
    y = a + b * x

    return y

def negfcn(x, *pars):
    fcn = pars[0]
    return -fcn(x, pars[1])


# Function that computes log-posterior
# given model parameters
def logpost(modelpars, lpinfo):
    # Model prediction
    ypred = lpinfo['model'](lpinfo['xd'], modelpars)
    # Data
    ydata = lpinfo['yd']
    nd = len(ydata)
    if lpinfo['ltype'] == 'classical':
        lpostm = 0.0
        for i in range(nd):
            for yy in ydata[i]:
                lpostm -= 0.5 * (ypred[i]-yy)**2/lpinfo['lparams']['sigma']**2
                lpostm -= 0.5 * np.log(2 * np.pi)
                lpostm -= np.log(lpinfo['lparams']['sigma'])
    else:
        print('Likelihood type is not recognized. Exiting')
        sys.exit()

    return lpostm

###
# Create synthetic data
###
# Define parameters for synthetic data generation
a = 1    # line intercept
b = 2    # line slope
sigma = 0.2  # std.dev. for data perturbations
npt = 13   # no. of data points
ncheck = 111
xmin = 1.0  # x-range min
xmax = 2.0  # x-range max


true_model = linear_model

# Uniformly random x samples
xd = xmin + (xmax - xmin) * np.random.rand(npt)
# Linear model for y samples
yd = true_model(xd, [a, b])
# Add noise
yd += sigma * np.random.randn(npt, )
yd = yd.reshape(-1,1)

xcheck = np.linspace(xmin, xmax, ncheck)
ycheck = true_model(xcheck, [a, b])





###
# Run MCMC
###


# Set the initial parameters and run MCMC
pdim = 2
zflag = True # start with an lbfgs
nmcmc = 10000  # number of MCMC samples requested
gamma = 0.1  # gamma parameter (jump size) of aMCMC
t0 = 100  # when adaptation, i.e. proposal covariance update, starts
tadapt = 100  # how often adaptive proposal covariance is updated

calib_model = linear_model



param_ini = np.random.rand(pdim)  # initial parameter values


# Set dictionary info for posterior computation
lpinfo = {'model': calib_model,
          'xd': xd, 'yd': [y for y in yd],
          'ltype': 'classical',
          'lparams': {'sigma': sigma}}


if zflag:
    res = minimize(negfcn, param_ini, args=(logpost,lpinfo), method='BFGS',options={'gtol': 1e-13})
    print('Opt:',res.x)
    param_ini = res.x


my_amcmc = AMCMC(t0=t0, tadapt=tadapt, gamma=gamma)
my_amcmc.setLogPost(logpost, None, lpinfo=lpinfo)
mcmc_results = my_amcmc.run(param_ini=param_ini, nmcmc=nmcmc)

samples, cmode, pmode = mcmc_results['chain'], mcmc_results['mapparams'], mcmc_results['maxpost']
np.savetxt('chain.txt', samples)


# Get MAP sample
ycheck_pred = calib_model(xcheck, cmode)
# Get MCMC prediction samples
ycheck_mcmc = np.empty((nmcmc, ncheck))
for i, sam in enumerate(samples[1:]):
    ycheck_mcmc[i, :] = calib_model(xcheck, sam)
ycheck_mean = np.average(ycheck_mcmc, axis=0)
ycheck_std = np.std(ycheck_mcmc, axis=0)

p, = plt.plot(xcheck, ycheck_mean, 'b-', label='Mean')
lc = lighten_color(p.get_color(), 0.5)
plt.fill_between(xcheck, ycheck_mean - ycheck_std,
                 ycheck_mean + ycheck_std, color=lc, zorder=-1000, alpha=1.0, label='StDev')


# Only if 1d
plt.plot(xcheck, ycheck_pred, 'g-', label='MAP')
plt.plot(xcheck, ycheck, 'k--', label='Truth')
plt.plot(xd, yd, 'ko', label='Data')
plt.legend()
plt.savefig('fitline.png')
