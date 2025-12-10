#!/usr/bin/env python
"""Example demonstrating MCMC-based Bayesian calibration of a multivariate linear model.

This script uses Adaptive MCMC to infer parameters of a linear model with
multiple features, including bias and weight parameters.
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
    y = par['b'] + x @ par['W']


    return y


def negfcn(x, *pars):
    fcn = pars[0]
    return -fcn(x, pars[1])


# Function that computes log-posterior
# given model parameters
def logpost(modelpars, lpinfo):
    # Model prediction
    ypred = lpinfo['model'](modelpars, lpinfo['modelpars'])
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
npt = 13   # no. of data points
pdim = 3
W = np.random.randn(pdim, npt)
b = np.random.randn(npt)
sigma = 0.2  # std.dev. for data perturbations
ncheck = 7

true_model_input = np.random.randn(pdim)

true_model, true_model_params = linear_model, {'W': W, 'b': b}
calib_model = linear_model

# from quinn.nns.nnwrap import nn_surrogate
# torch.set_default_tensor_type(torch.DoubleTensor)
# nnlin=torch.nn.Linear(pdim, npt)
# true_model, true_model_params = nn_surrogate, nnlin
# calib_model = nn_surrogate


yd = true_model(true_model_input, true_model_params)
# Add noise
neach = 5
yd = np.tile(yd, (neach,1)).T
yd += sigma * np.random.randn(npt, neach)
#yd = yd.reshape(-1,1)



###
# Run MCMC
###


# Set the initial parameters and run MCMC
zflag = True
nmcmc = 10000  # number of MCMC samples requested
gamma = 0.1  # gamma parameter (jump size) of aMCMC
t0 = 100  # when adaptation, i.e. proposal covariance update, starts
tadapt = 100  # how often adaptive proposal covariance is updated




param_ini = np.random.rand(pdim)  # initial parameter values


# Set dictionary info for posterior computation
lpinfo = {'model': calib_model,
          'modelpars': true_model_params, 'yd': [y for y in yd],
          'ltype': 'classical',
          'lparams': {'sigma': sigma}}


if zflag:
    res = minimize(negfcn, param_ini, args=(logpost,lpinfo), method='BFGS',options={'gtol': 1e-13})
    print('Opt:', res.x)
    param_ini = res.x

my_amcmc = AMCMC(t0=t0, tadapt=tadapt, gamma=gamma)
my_amcmc.setLogPost(logpost, None, lpinfo=lpinfo)
mcmc_results = my_amcmc.run(param_ini=param_ini, nmcmc=nmcmc)

samples, cmode, pmode = mcmc_results['chain'], mcmc_results['mapparams'], mcmc_results['maxpost']
np.savetxt('chain.txt', samples)


print("True values :", true_model_input)
print("MAP values  :", cmode)

# Get MAP sample
ycheck_pred = calib_model(cmode, true_model_params)
# Get MCMC prediction samples
ycheck_mcmc = np.empty((nmcmc, npt))
for i, sam in enumerate(samples[1:]):
    ycheck_mcmc[i, :] = calib_model(sam, true_model_params)
ycheck_mean = np.average(ycheck_mcmc, axis=0)
ycheck_std = np.std(ycheck_mcmc, axis=0)

p, = plt.plot(np.arange(npt), ycheck_mean, 'b-', label='Mean')
lc = lighten_color(p.get_color(), 0.5)
plt.fill_between(np.arange(npt), ycheck_mean - ycheck_std,
                 ycheck_mean + ycheck_std, color=lc, alpha=1.0, label='StDev')


plt.plot(np.arange(npt), ycheck_pred, 'g-', label='MAP')
plt.plot(np.arange(npt), yd, 'ko', label='Data')
h, l = plt.gca().get_legend_handles_labels()
# indexes = [l.index(x) for x in set(l)]
# h, l = [h[i] for i in indexes], [l[i] for i in indexes]
h, l = [h[-2]]+[h[0]]+[h[-1]]+[h[1]], [l[-2]]+[l[0]]+[l[-1]]+[l[1]]
plt.legend(h, l)
plt.savefig('fitmodel.png')
