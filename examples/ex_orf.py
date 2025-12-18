#!/usr/bin/env python
"""Example for testing orthonormalization of functions using Gram-Schmidt or QR decomposition.

Written by Habib N. Najm (2025).
"""

import sys
import argparse
import numpy as np
import copy
import matplotlib.pyplot as plt

import pytuq.ftools.orf as orf

np.set_printoptions(precision=6, linewidth=200, suppress=False, threshold=np.inf)

###############################################################################
###############################################################################
###############################################################################
# define some options
#Method     (str)  : specifies if you want to use QR ('QR') or Gram-Schmidt ('GS')
#modified   (bool) : specifies modified GS, or not (ignored if using QR)
#nstage     (int)  : number of stages, > 0 (ignored if using QR)
#verbose    (bool) : specifies if verbose or not
#ndim       (int)  : specifies test case dimensionality, 1 or 2 
#plot       (bool) : spefies if you want to make plots or not
parser = argparse.ArgumentParser(description="ex_orf arguments")

parser.add_argument('--Method'  , '-M', type   = str,          default = 'QR',  help = "specify orthonormalization method ('QR') or ('GS').")
parser.add_argument('--modified', '-m', action = 'store_true', default = False, help = 'True/False as in modified or not (ignored with QR.')
parser.add_argument('--nstage'  , '-s', type   = int,          default = 1,     help = 'Specify number of stages (ignored with QR).')
parser.add_argument('--verbose' , '-v', action = 'store_true', default = False, help = 'Enable verbose output.')
parser.add_argument('--ndim'    , '-n', type   = int,          default = 1,     help = 'Specify independent data point dimensionality.')
parser.add_argument('--plot'    , '-p', action = 'store_true', default = False, help = 'Enable plotting.')
args = parser.parse_args()

wrnd = {'QR':'(ignored)','GS':''}
print('Method:   ',args.Method)
print('modified: ',args.modified, wrnd[args.Method])
print('nstage:   ',args.nstage, wrnd[args.Method])
print('verbose:  ',args.verbose)
print('ndim:     ',args.ndim)
print('plot:     ',args.plot)

Method   = args.Method
modified = args.modified
nstage   = args.nstage
verbose  = args.verbose
ndim     = args.ndim
plot     = args.plot

# numpy rng initialization
rng       = np.random.default_rng(345126)     
np.random.seed(345126)

###############################################################################
###############################################################################
###############################################################################

#==============================================================================
# specify the set of starting functions phi0, with shape (ntrm,)
#   and the data x, with shape (npt,ndim), and 
# npt is the number of data points, where each data point is ndim dimensions
#   in general can use any ndim as long as the appropriate phi0 functions are set up
#   here we have two example options, with ndim = 1 or 2

if ndim == 1:    # 1d data case

    # function to return 1d monomial phi[k](x) functions
    def phif_1d(k):
        def lfnc(x):
            return x**k
        return lfnc

    # set number of terms: phi_0(x), phi_1(x), ... with phi_k(x) = x**k per above phif_1d()
    ntrm = 10
    phi0 = np.array([phif_1d(k) for k in range(ntrm)])

    # define 1d data, an npt-long 1d array of points, each a scalar value
    # sampled uniformly on [-1,1]
    npt  = 100
    x    = rng.uniform(-1,1,(npt))

elif ndim == 2:    # 2d data case

    # function to return 2d tensor-product of monomials phi[k0,k1](x) functions
    # where x is a 2-vector
    def phif_2d(k0,k1):
        def lfnc(x_):
            return np.array([x[0]**k0 * x[1]**k1 for x in x_])
        return lfnc

    ntrm0 = 4
    ntrm1 = 4
    ntrm  = ntrm0*ntrm1
    phi0  = np.array([phif_2d(k0,k1) for k0 in range(ntrm0) for k1 in range(ntrm1)])

    # define 2d data, an npt x 2 array of npt points, each being a 2-vector
    # each component of which is sampled uniformly on [-1,1]
    npt  = 100
    x    = rng.uniform(-1,1,(npt,2))

else:
    print('not a valid ndim case option')
    sys.exit(1)

print('ntrm =',ntrm)

#==============================================================================
# specify the linear map, which will also hold the x data

lmap = orf.GLMAP(x, verbose=verbose)                                        # identity map
# lmap = orf.GLMAP(x, 10*np.random.rand(int(npt/2),npt),verbose=verbose)    # user matrix map
# lmap = orf.GLMAP(x, lambda x, u: np.exp(x) * u, verbose=verbose)          # user function map

#==============================================================================
# do orthonormalization

# copy starting array of functions
phi  = copy.deepcopy(phi0)

if Method == 'QR':
    # orthonormalize with QR factorization
    qr        = orf.QR(phi,lmap)
    Pmat, tht = qr.ortho(verbose=verbose)
    print('ortho check phi0 maxabs:',np.max(np.abs(qr.ortho_check_phi() - np.eye(ntrm))))
    print('ortho check tht  maxabs:',np.max(np.abs(qr.ortho_check()     - np.eye(ntrm))))
elif Method == 'GS':
    # orthonormalize with multistage modified GS
    mmgs      = orf.MMGS(phi,lmap)
    Pmat, tht = mmgs.ortho(modified=modified,nstage=nstage,verbose=verbose)
    print('ortho check phi0 maxabs:',np.max(np.abs(mmgs.ortho_check_phi(0)    - np.eye(ntrm))))
    print('ortho check tht  maxabs:',np.max(np.abs(mmgs.ortho_check(nstage-1) - np.eye(ntrm))))
else:
    print("Only Method = 'QR' or 'GS' is allowed.")
    sys.exit(1)

#==============================================================================
# plotting diagnostics
if plot:

    if ndim == 1:   

        nx    = 250
        xt    = np.linspace(-1,1,nx)
        phixt = np.array([f(xt) for f in phi]).T
        thtxt = np.array([f(xt) for f in tht]).T
        for q in range(ntrm):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            ax1.plot(xt, phixt[:,q], label='k:'+str(q))
            ax2.plot(xt, thtxt[:,q], label='k:'+str(q))
            ax1.set_title(r'$\phi_{'+str(q)+'}(x)$')
            ax2.set_title(r'$\theta_{'+str(q)+'}(x)$')
            for ax in (ax1, ax2):
                ax.set_xlabel(r'$x$')
                ax.grid(True)
            ax1.set_ylabel(r'$\phi_{'+str(q)+'}(x)$')
            ax2.set_ylabel(r'$\theta_{'+str(q)+'}(x)$')
            ymn = min(np.min(phixt),np.min(thtxt))
            ymx = max(np.max(phixt),np.max(thtxt))
            eps = 0.05*(ymx-ymn)
            ax1.set_ylim(ymn-eps,ymx+eps)
            ax2.set_ylim(ymn-eps,ymx+eps)
            plt.savefig(f'orf_1d_{q}.png')
            #plt.show()

    elif ndim == 2:

        X    = np.arange(-2, 2, 0.1)
        Y    = np.arange(-1, 1, 0.05)
        nX   = X.shape[0]
        nY   = Y.shape[0]
        X, Y = np.meshgrid(X, Y)
        XY   = np.vstack((X.flatten(),Y.flatten())).T

        Zphi = np.array([phi[q](XY).reshape(nX,nY) for q in range(ntrm)])
        Ztht = np.array([tht[q](XY).reshape(nX,nY) for q in range(ntrm)])

        for q in range(ntrm):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
            ax1.plot_surface(X, Y, Zphi[q], cmap='viridis')
            ax2.plot_surface(X, Y, Ztht[q], cmap='viridis')
            ax1.set_title(r'$\phi_{'+str(q)+'}(x,y)$')
            ax2.set_title(r'$\theta_{'+str(q)+'}(x,y)$')
            for ax in (ax1,ax2):
                ax.set_xlabel(r'$x$')
                ax.set_ylabel(r'$y$')
            ax1.set_zlabel(r'$\phi_{'+str(q)+'}(x,y)$')
            ax2.set_zlabel(r'$\theta_{'+str(q)+'}(x,y)$')
            zmn = min(np.min(Zphi[q]),np.min(Ztht[q]))
            zmx = max(np.max(Zphi[q]),np.max(Ztht[q]))
            eps = 0.05*(zmx-zmn)
            ax1.set_zlim(zmn-eps,zmx+eps)
            ax2.set_zlim(zmn-eps,zmx+eps)
            plt.savefig(f'orf_2d_{q}.png')
            #plt.show()
    else:
        print('not a valid ndim case option')
        sys.exit(1)
#==============================================================================

print('done')

