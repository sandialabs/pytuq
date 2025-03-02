#!/usr/bin/env python

"""Example demonstrating joint PC construction given samples.

Uses Rosenblatt transformation and joint regression in parameteric/stochastic space.
"""

import numpy as np
import matplotlib.pyplot as plt

from pytuq.lreg.anl import anl
from pytuq.rv.pcrv import PCRV
from pytuq.rv.rosen import Rosenblatt
from pytuq.utils.mindex import get_mi
import pytuq.utils.plotting as xp

# Plotting settings
xp.myrc()

#############################################
#############################################
#############################################

# An example with param dim. = 2 and stoch. dim. = 2
def sample_model_d2_s2(par, n_resam):
    npar, pdim = par.shape
    assert(pdim==2)
    a=par[:, 0]
    b=par[:, 1]

    out = np.empty((npar, n_resam, 2))

    xisam = np.random.randn(n_resam, 2)
    out[:,:,0] = a.reshape(-1,1)**2 * (xisam[:,0]).reshape(1,-1)

    out[:,:,1] = a.reshape(-1,1)*(xisam[:,0]**2).reshape(1,-1) + \
                 b.reshape(-1,1)*xisam[:,1].reshape(1,-1)


    return out

# An example with param dim. = 1 and stoch. dim. = 1
def sample_model_d1_s1(par, n_resam):
    npar, pdim = par.shape
    assert(pdim==1)
    a=par[:, 0]

    out = np.empty((npar, n_resam, 1))

    xisam = 2.*np.random.rand(n_resam, 1)-1.
    out[:,:,0] = a.reshape(-1,1) * (xisam[:,0]).reshape(1,-1)


    return out

#############################################
#############################################
#############################################

## Sampling function

# 1+1 dim example
sample_model = sample_model_d1_s1
# Parameter range
param_range = np.array([[-1,1]])
# Stoch. dimensionality
sdim = 1

# # 2+2 dim example
# sample_model = sample_model_d2_s2
# # Parameter range
# param_range = np.array([[-1,1], [0,3]])
# # Stoch. dimensionality
# sdim = 2

# Number of samples in parametric space (ensemble size)
npar = 22
# Number of samples in stochastic space (replica size)
nsam = 112

# Number of re-samples (for validation)
n_resam = 44

# Extract parameter dimensionality
pdim, two = param_range.shape
assert(two==2)

# Sample parameters in [-1, 1] and in physical range
tmp = np.random.rand(npar, pdim)
param_samples = 2.*tmp-1.
param_samples_phys = tmp*(param_range[:,1]-param_range[:,0])+param_range[:,0]

# Generate output samples for Rosenblatt
# for each parameter sample, generate nsam 'seed' or random replica samples
model_samples = sample_model(param_samples_phys, nsam)
npar_, nsam_, odim = model_samples.shape

# Sanity checks on dimesnionalities
assert(npar_==npar)
assert(nsam_==nsam)
assert(sdim == odim)


colors = xp.set_colors(npar)

all_cfs=[]
# For each parameter sample, evaluate inverse Rosenblatt on that particular 'slice'
for ipar, param_sample in enumerate(param_samples):
    ros = Rosenblatt(model_samples[ipar, :, :])
    # PC order
    order = 1


    # Setup the PC
    mindex = get_mi(order, sdim)
    pcrv = PCRV(odim, sdim, 'LU', mi=mindex)

    # Create regression input/output data 
    reg_input = np.empty((n_resam, sdim))
    reg_output = np.empty((n_resam, odim))

    unif_sam = np.zeros((n_resam, sdim))
    for idim in range(sdim):
        reg_input[:, idim] = pcrv.PC1ds[idim].germSample(n_resam)
        unif_sam[:, idim] = pcrv.PC1ds[idim].germCdf(reg_input[:, idim])
    reg_output = np.array([ros.inv(u) for u in unif_sam])

    # Plot regression data for all parameters
    for isdim in range(sdim):
        for iodim in range(sdim):
            _=plt.figure(figsize=(10,9))
            indx = np.argsort(reg_input[:, isdim])
            plt.plot(reg_input[indx, isdim],
                         reg_output[indx, iodim], 'o-', color=colors[ipar])
            plt.xlabel(f'Germ $\\xi_{isdim+1}$')
            plt.ylabel(f'R.v.  $X_{iodim+1}$')
            plt.savefig(f'iros_s{isdim}_o{iodim}_p{ipar}.png')
            plt.clf()



    # Diagonal plot to check the accuracy of regression
    cfs=[]
    for idim in range(odim):
        Amat = pcrv.evalBases(reg_input, idim)
        lreg = anl()
        lreg.fita(Amat, reg_output[:, idim])
        cfs.append(lreg.cf)
        xp.plot_dm([reg_output[:, idim]], [lreg.predicta(Amat)[0]],
                    figname=f'dm_o{idim}.png',
                    axes_labels=['InvRos Data', 'PC Apprx'],
                    labels=['Training'],
                    msize=9)

    # Set the learnt PC coefficients
    pcrv.setCfs(cfs)
    all_cfs.append(cfs)

    ## Validate PDFs on training parameter values
    # Re-sample amount for each parameter value
    n_resam = 111
    input_pc = np.zeros((n_resam, sdim))
    input_pc[:, :pdim] = param_samples[ipar, :]
    for idim in range(sdim):
        input_pc[:, idim] = pcrv.PC1ds[idim].germSample(n_resam)
    output_pc = pcrv.evalPC(input_pc)
    xp.plot_samples_pdfs([model_samples[ipar, :, :], output_pc],
                  legends=['Model Samples', 'PC Samples'],
                  colors=['b', 'r'],
                  title=f'Par = {param_samples_phys[ipar]}',
                  file_prefix=f'xpdfpc_p{ipar}_t')



all_cfs_ = np.array(all_cfs)
npar__, odim_, npc = all_cfs_.shape
assert(npar__==npar)
assert(odim==odim)

cols = npc
rows = odim
ypad = 0.3

for idim in range(pdim):
    #indx = np.argsort(param_samples_phys[:, idim])

    fig, axes = plt.subplots(rows, cols, figsize=(8*cols,(8+ypad)*rows),
                                 gridspec_kw={'hspace': ypad, 'wspace': 0.3})
    for iout in range(odim):
        for ipc in range(npc):
            axes=axes.reshape(rows, cols)
            axes[iout, ipc].plot(param_samples_phys[:, idim], all_cfs_[:, iout, ipc], 'o')
            axes[iout, ipc].set_xlabel(rf'$p_{idim}$')
            axes[iout, ipc].set_ylabel('$cf_{'+f'{iout},{ipc}'+'}$')
    plt.savefig(f'pcf_p{idim}.png')
    plt.clf()


# ## Validate PDFs on other (testing) parameter values
# # Randomly select some parameter values
# npar_valid = 5
# tmp = np.random.rand(npar_valid, pdim)
# param_samples_valid = 2.*tmp-1.
# param_samples_valid_phys = tmp*(param_range[:,1]-param_range[:,0])+param_range[:,0]
# # Re-sample amount for each parameter value
# n_resam_valid = 111
# # Evaluate model samples for validation
# model_valid = sample_model(param_samples_valid_phys, n_resam_valid)

# for ipar in range(npar_valid):
#     input_pc_valid = np.zeros((n_resam_valid, sdim))
#     input_pc_valid[:, :pdim] = param_samples_valid[ipar, :]
#     for idim in range(sdim):
#         input_pc_valid[:, pdim+idim] = pcrv.PC1ds[pdim+idim].germSample(n_resam_valid)
#     output_pc_valid = pcrv.evalPC(input_pc_valid)
#     xp.plot_samples_pdfs([model_valid[ipar, :, :], output_pc_valid],
#                   legends=['Model Samples', 'PC Samples'],
#                   colors=['b', 'r'],
#                   title=f'Par = {param_samples_valid_phys[ipar]}',
#                   file_prefix=f'xpdfpc_p{ipar}_v')
