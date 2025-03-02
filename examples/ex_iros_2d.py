#!/usr/bin/env python

"""Example demonstrating Rosenblatt transformation."""

import numpy as np
import matplotlib.pyplot as plt

from pytuq.rv.rosen import Rosenblatt
import pytuq.utils.plotting as xp

# Plotting settings
xp.myrc()

#############################################
#############################################
#############################################

# Sampling model
def mvn(nsam, ndim):
    assert(ndim==2)
    mean = np.array([-1.,2.])
    cov = np.eye(2)
    cov[0,1]=cov[1,0]=0.8
    xsam = np.random.multivariate_normal(mean, cov, size=(nsam,))
    return xsam

# Sampling model
def expu(nsam, ndim):

    return np.exp(np.random.rand(nsam, ndim))

# Sampling model
def unif(nsam, ndim):

    return 2.*np.random.rand(nsam, ndim)+1.

#############################################
#############################################
#############################################

# Number of original samples used to construct Rosenblatt map
nsam = 11
# Dimensionality of random variable
ndim = 2
# Sampling function
sampling_true = expu #unif #mvn #expu

ngr = 33


# Original samples that are used to construct Rosenblatt map
xsam = sampling_true(nsam, ndim)

ros = Rosenblatt(xsam)#, sigmas=0.2*np.ones(ndim))


####
#### Plot forward Rosenblatt function
####
ygr1 = np.linspace(0., 4., ngr)
ygr2 = np.linspace(0., 4., ngr)
X, Y = np.meshgrid(ygr1, ygr2)
ygr = np.vstack((X.flatten(), Y.flatten())).T
fros_ygr = np.array([ros(x) for x in ygr])

for idim in range(ndim):
    #plt.contour(X, Y, fros_ygr[:,idim].reshape(X.shape), 22, linewidths=1)
    for jgr in range(ngr):
        plt.plot(ygr1, fros_ygr[jgr*ngr:(jgr+1)*ngr,idim])
    plt.xlabel('$x_0$')
    plt.ylabel(f'$u_{idim}$')
    plt.savefig(f'fros_u{idim}_x0.png')
    plt.clf()

    for jgr in range(ngr):
        plt.plot(ygr2, fros_ygr[jgr::ngr,idim])
    plt.xlabel('$x_1$')
    plt.ylabel(f'$u_{idim}$')
    plt.savefig(f'fros_u{idim}_x1.png')
    plt.clf()


####
#### Plot inverse Rosenblatt function
####

# Sample new uniform r.v.
xgr1 = np.linspace(0.00001, 1.-0.00001, ngr)
xgr2 = np.linspace(0.00001, 1.-0.00001, ngr)
X, Y = np.meshgrid(xgr1, xgr2)
xgr = np.vstack((X.flatten(), Y.flatten())).T
iros_xgr = np.array([ros.inv(u) for u in xgr])

for idim in range(ndim):
    #plt.contour(X, Y, iros_xgr[:,idim].reshape(X.shape), 22, linewidths=1)
    for jgr in range(ngr):
        plt.plot(xgr1, iros_xgr[jgr*ngr:(jgr+1)*ngr,idim])
    plt.xlabel('$u_0$')
    plt.ylabel(f'$x_{idim}$')
    plt.savefig(f'iros_x{idim}_u0.png')
    plt.clf()

    for jgr in range(ngr):
        plt.plot(xgr2, iros_xgr[jgr::ngr,idim])
    plt.xlabel('$u_1$')
    plt.ylabel(f'$x_{idim}$')
    plt.savefig(f'iros_x{idim}_u1.png')
    plt.clf()



