#!/usr/bin/env python


import sys
import numpy as np
import matplotlib.pyplot as plt

from pytuq.rv.pcrv import PCRV
from pytuq.linred.kle import KLE
from pytuq.workflows.fits import pc_fit
from pytuq.utils.plotting import myrc, plot_sens, plot_dm
from pytuq.utils.mindex import micf_join

myrc()


####################################################################
####################################################################

# Number of samples (N), dimensionality of input (d), dimensionality of output (M)
nsam, ndim, nout = 111, 5, 33
# Input parameter
ptrain = 2.*np.random.rand(nsam, ndim)-1
# Toy model: f_i = \exp(\sum_{j=1}^d a_{ij} p_j) for some random a_{ij}, for i=1,..., M.
ytrain = np.exp(np.dot(ptrain, np.random.rand(ndim, nout)))


## ... OR read from files (ptrain needs to be in [-1,1])
# ptrain = np.loadtxt('ptrain.txt') # N x d
# ytrain = np.loadtxt('ytrain.txt') # N x M
# if len(ytrain.shape)==1:
#     ytrain = ytrain[:, np.newaxis]
# nsam, nout = ytrain.shape
# if len(ptrain.shape)==1:
#     ptrain = ptrain[:, np.newaxis]
# nsam_, ndim = ptrain.shape
# assert(nsam==nsam_)

## Build KLE
kl = KLE()
kl.build(ytrain.T, plot=True)
## Get the number of eigenvalues that capture 99% of variance
neig = kl.get_neig(0.99)
## Evaluate the truncated KL up to the selected number of eigenvalues
ytrain_kl = kl.eval(neig=neig)
## Plot explained variance
kl.plot_expvar()

## Get the eigenfeatures to build surrogates for
xitrain = kl.xi[:, :neig]

## Fit a PC to these eigenfeatures
# pcrv = pc_fit(ptrain, xitrain, order=3, pctype='LU', method='bcs', bcs_eta=1.e-8)
pcrv = pc_fit(ptrain, xitrain, order=3, pctype='LU', method='lsq')
## Evaluate PC surrogate in eigen-space
xitrain_kl = pcrv.function(ptrain)
## Plug-in to KLE to get the 'physical' approximation
ytrain_klpc = kl.eval(xi=xitrain_kl, neig=neig)


## Create a PC object with common multiindex to get the sensitivities for all model outputs
mindex_all, cfs_all = micf_join(pcrv.mindices, pcrv.coefs)
cfs_glob = np.dot(np.dot(cfs_all.T, np.diag(np.sqrt(kl.eigval[:neig]))), kl.modes[:, :neig].T) #npc, nout
cfs_glob[0, :] += kl.mean

print("Multiindex shape :", mindex_all.shape)
print("PC coefficients' shape :", cfs_glob.shape) #npc, nout
nx = cfs_glob.shape[1]
pcrv_phys=PCRV(nout, ndim, pcrv.pctypes, mi=mindex_all, cfs=cfs_glob.T)

## Compute sensitivities
allsens = pcrv_phys.computeTotSens()

## Plot sensitivities
pars = range(ndim)
cases = range(nout)
plot_sens(allsens,pars,cases,vis="bar",ncol=5,
          par_labels=[rf'p$_{i}$' for i in cases], case_labels=[r'qoi$_{'+f'{i}'+'}$' for i in cases],lbl_size=25, xticklabel_size=18, legend_size=18, xticklabel_rotation=90,
          figname='sens_klpc.png')


## Plot the model and its approximations (for every 20th sample to avoid too many figures)
for isam in range(0, nsam, 20):
    f = plt.figure(figsize=(12,4))
    plt.plot(range(nout), ytrain[isam, :], 'bo-', ms=8, label='Model')
    plt.plot(range(nout), ytrain_kl.T[isam, :], 'go-', ms=8, label=f'KL')
    plt.plot(range(nout), ytrain_klpc.T[isam, :], 'mo-', ms=8, label=f'KLPC')
    plt.xlabel('Output ID')
    plt.ylabel('QoI')
    plt.legend()
    plt.title(f'Sample #{isam+1}/{nsam}')
    plt.xticks(range(nout))
    plt.tight_layout()
    plt.savefig(f'fit_s{str(isam+1).zfill(3)}.png')
    #plt.savefig(f'fit_e{str(neig).zfill(3)}_s{str(isam).zfill(3)}.png')
    plt.close(f)


## Parity plots of model and its KLPC approximation (for every 10th output to avoid too many figures)
for iout in range(0, nout, 10):
    plot_dm([ytrain[:,iout]], [ytrain_klpc[iout,:]], errorbars=None, labels=['Training'], colors=None, axes_labels=['Model', 'KLPC Apprx.'], figname=f'dm_{str(iout+1).zfill(3)}.png', legendpos='in', msize=7)
    plt.close(plt.gcf())
