#!/usr/bin/env python

import sys
import torch
import numpy as np


import pytuq.minf.minf as minf
import pytuq.gsa.gsa as gsa

try:
    from quinn.nns.nnwrap import nn_p, nnwrapper
except ImportError:
    print("Warning: QUiNN not installed. NN functionality won't work.")

#torch.set_default_tensor_type(torch.DoubleTensor)



def quad_model_single(par, xx):
    return par[0] + par[1] * xx +  par[2] * xx**2

def quad_model(pp, xx):
    yy = np.empty((pp.shape[0], xx.shape[0]))
    for i in range(len(pp)):
        yy[i, :] = quad_model_single(pp[i, :], xx).T
    return yy


def nn_model(pp, xx):
    nnlin = torch.nn.Linear(1, 1)
    yy = np.empty((pp.shape[0], xx.shape[0]))
    for i in range(len(pp)):
        yy[i, :] = nn_p(pp[i,:], xx.reshape(-1, 1), nnlin)[:,0]
    return yy


def nn_model_surrogate(pp, xx):
    nout = xx.shape[0]
    nin = pp.shape[1]
    #nnmodel = torch.nn.Linear(nin, nout)

    nnmodel = torch.nn.Sequential(torch.nn.Linear(nin, 20), torch.nn.Tanh(),
                                  torch.nn.Linear(20, 20), torch.nn.Tanh(),
                                  torch.nn.Linear(20, nout))

    np.random.seed(133)
    npar = sum(p.numel() for p in nnmodel.parameters())
    ww = np.random.rand(npar,)
    yy = nn_p(ww, pp, nnmodel)
    return yy

###
# Create synthetic data
###
data_sigma = 0.05  # std.dev. for data perturbations
npt = 13   # no. of data points
# Uniformly random x samples
xd = np.arange(npt)/npt

true_model, true_model_params = quad_model_single, xd


# True model for y samples
yd_true = true_model(np.array([1., 3., -2.5]), true_model_params)
# Add noise
neach = 5
#yd[5]+=57.0
yd = np.tile(yd_true, (neach,1)).T
yd += data_sigma * np.random.randn(npt, neach)
#yd = yd.reshape(-1,1)



# calib_model, calib_model_params = nn_model_surrogate, xd
# pdim = 5

# calib_model, calib_model_params = nn_model, xd
# pdim = 2

calib_model, calib_model_params = quad_model, xd
pdim = 3


checkmode = [len(xd), xd, range(len(xd))]

# Do sensitivities just in case
sens_main, sens_tot = gsa.model_sens(calib_model, calib_model_params,
                                 np.tile(np.array([-1., 1.]), (pdim, 1)),
                                 method='SamSobol', nsam=1000, plot=True)

ind_calib = [1, 4, 11] #range(npt) #[3, 8, 10]

calib_params={'param_ini': np.random.rand(pdim), 'cov_ini': None,
't0': 100, 'tadapt' : 100, 'gamma' : 0.1, 'nmcmc' : 1000}

## Model parameter inference configuration
model_infer_config = {
    "inpdf_type" : 'pci', "pc_type" : 'LU', "outord" : 0,
    "rndind" : [], "fixindnom" : [], "ind_calib" : ind_calib,
    "calib_type" : 'amcmc', "calib_params" : calib_params,
    "md_transform" : None, "zflag" : True, "datamode" : None,
    "lik_type" : 'classical', "lik_params" : {},
    "pr_type" : 'uniform', "pr_params" : {},
    "dv_type" : 'var_fixed', "dv_params" : [data_sigma**2]
}

## Beautified printing
try:
    pprint.pprint(model_infer_config, indent=1, compact=True)
except NameError:
    print(model_infer_config)

## Run the inference
calib_results = minf.model_infer(yd[ind_calib],
                                 calib_model, calib_model_params, pdim,
                                 **model_infer_config)


np.savetxt('chain.txt', calib_results['chain'])

## Postprocess model at grid points for plotting
checkmode = [calib_results['post'].model, xd, range(len(xd)), None]
ycheck_pred, samples =  minf.model_infer_postp(calib_results,
                                               checkmode=checkmode,
                                               nburn=len(calib_results['chain'])//10,
                                               nevery=10, nxi=1)


psamples, fsamples = samples['pmcmc'], samples['fmcmc']

## More plots to visualize fits
minf.plot_1dfit(xd, ycheck_pred, ygrid_true=None, xydata=(xd,yd))

