#!/usr/bin/env python

"""Example demonstrating PC construction given samples.

Uses Rosenblatt transformation and regression.
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

# Number of original samples
nsam = 333
# Number of PC re-samples
n_pcsam = 1111
# Stochastic dimensionality
sdim = 2
# PC order
order = 4
# Random variable dimensionality
odim = 2
# Unless we know how sampling is done, we always take sdim==odim
assert(odim==sdim)
# Create the PC object
pcrv = PCRV(odim, sdim, 'LU', mi=get_mi(order, sdim))

# Create a Rosenblatt map
xsam = np.exp(np.random.rand(nsam, odim))
ros = Rosenblatt(xsam)


# Sample uniform through the germ
nreg = 77
germ_sam = pcrv.sampleGerm(nreg)
unif_sam = np.zeros((nreg, sdim))
for idim in range(sdim):
    unif_sam[:, idim] = pcrv.PC1ds[idim].germCdf(germ_sam[:, idim])

# Evaluate inverse Rosenblatt as regression training data
xreg = np.array([ros.inv(u) for u in unif_sam])

# Regression fit of inverse Roesnblatt function
# Also plots the training data with the PC fit
all_cfs=[]
for idim in range(odim):
    Amat = pcrv.evalBases(germ_sam, idim)
    lreg = anl()
    lreg.fita(Amat, xreg[:, idim])
    all_cfs.append(lreg.cf)
    _ = plt.figure(figsize=(10,8))
    plt.plot(germ_sam[:, idim], xreg[:, idim], 'bo', label=r'X=R$^{-1}$(U) regression data')
    ngr = 111
    f = 1.0
    xgr = np.linspace(f*np.min(germ_sam[:, idim]), f*np.max(germ_sam[:, idim]), ngr)
    xgr0 = np.zeros((ngr, sdim)) # slice at 0
    xgr0[:, idim]= xgr
    Agr = pcrv.evalBases(xgr0, idim)
    ss = ''
    for ii in range(idim):
        ss += f'\\xi_{ii+1}=0, '
    plt.plot(xgr, lreg.predicta(Agr)[0], 'g-', label=f'Fitted PC slice $X^{{PC}}_{idim+1}('+ss+f'\\xi_{idim+1})$', zorder=-1000)
    plt.xlabel(f'Germ $\\xi_{idim+1}$')
    plt.ylabel(f'R.v.  $X_{idim+1}$')
    plt.legend()
    plt.savefig(f'iros_d{idim}.png') # Plotting is really useful for the first dimension. For the others, it is merely a slice.
    plt.clf()

# Resample PC
pcrv.setCfs(all_cfs)
pcsam = pcrv.sample(n_pcsam)
np.savetxt('pcsam.txt', pcsam)

# Plot PDFs for comparison
xp.plot_samples_pdfs([xsam, xreg, pcsam],
                  legends=['Orig. Samples', r'R$^{-1}$(U) Samples', 'PC Samples'],
                  colors=['b', 'g', 'r'],
                  file_prefix='xpdfpc')

