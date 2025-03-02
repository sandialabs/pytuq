#!/usr/bin/env python
"""Run script testing PC construction given samples.

Uses Rosenblatt transformation and regression.
"""

import numpy as np
import matplotlib.pyplot as plt


from pytuq.utils.plotting import myrc
from pytuq.workflows.fits import pc_ros

myrc()


#############################################
#############################################
#############################################

nrepl = 10 # Number of repeated tests to judge any systematic issues
nsam = 111 # Number of original samples used to build Rosenblatt map
nreg = 77 # Number of samples for regression to build PC
ndim = 1
order = 1 # PC order
# Bandwidth factor for Rosenblatt map. 1.0 means we take the optimal
bwfactor = 1.0

#############################################
#############################################
#############################################

sample_mean = np.empty((ndim, nrepl))
sample_std = np.empty((ndim, nrepl))
pc_mean = np.empty((ndim, nrepl))
pc_std = np.empty((ndim, nrepl))
for ir in range(nrepl):
    print(f"Replica {ir+1}/{nrepl}")
    xsam = np.random.randn(nsam, ndim)
    pcrv = pc_ros(xsam, pctype='HG', order=order,
                  nreg=nreg, bwfactor=bwfactor)

    sample_mean[:, ir], sample_std[:, ir] = np.mean(xsam, axis=0), np.std(xsam, axis=0)
    for idim in range(ndim):
        pc_mean[idim, ir], pc_std[idim, ir] = pcrv.computeMean()[idim], np.sqrt(pcrv.computeVar()[idim])


for idim in range(ndim):
    plt.figure(figsize=(12,8))
    plt.plot(range(nrepl), sample_mean[idim], 'bo', label='Sample mean')
    plt.plot(range(nrepl), pc_mean[idim], 'ro', label='PC mean')
    plt.plot(range(nrepl), np.zeros(nrepl), 'g--', label='True mean')
    plt.legend()
    plt.savefig(f'means_d{idim}.png')
    plt.clf()

    plt.figure(figsize=(10,10))
    a = min(np.min(sample_mean[idim]), np.min(pc_mean[idim]))
    b = max(np.max(sample_mean[idim]), np.max(pc_mean[idim]))
    plt.plot([a,b], [a,b], 'g--')
    plt.plot(sample_mean[idim], pc_mean[idim], 'bo')
    plt.xlabel('Sample mean')
    plt.ylabel('PC mean')
    plt.savefig(f'means_diag_d{idim}.png')
    plt.clf()

    plt.figure(figsize=(12,8))
    plt.plot(range(nrepl), sample_std[idim], 'bo', label='Sample stdev')
    plt.plot(range(nrepl), pc_std[idim], 'ro', label='PC stdev')
    plt.plot(range(nrepl), np.ones(nrepl), 'g--', label='True stdev')
    plt.legend()
    plt.savefig(f'stdevs_d{idim}.png')
    plt.clf()

    plt.figure(figsize=(10,10))
    a = min(np.min(sample_std[idim]), np.min(pc_std[idim]))
    b = max(np.max(sample_std[idim]), np.max(pc_std[idim]))
    plt.plot([a,b], [a,b], 'g--')
    plt.plot(sample_std[idim], pc_std[idim], 'bo')
    plt.xlabel('Sample stdev')
    plt.ylabel('PC stdev')
    plt.savefig(f'stdevs_diag_d{idim}.png')
    plt.clf()
