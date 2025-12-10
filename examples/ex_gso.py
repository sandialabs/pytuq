#!/usr/bin/env python
"""Example for testing Multistage Modified Gram-Schmidt (MMGS) orthogonalization for functions.

Written by Habib N. Najm (2025).
"""

import sys
import argparse
import numpy as np
import copy
import matplotlib.pyplot as plt

import pytuq.ftools.gso as gso

np.set_printoptions(precision=6, linewidth=200, suppress=False, threshold=np.inf)

###############################################################################
###############################################################################
###############################################################################
# define some options
#modified   (bool) : specifies modified GS, or not
#nstage     (int)  : number of stages, > 0 
#verbose    (bool) : specifies if verbose or not
#ndim       (int)  : specifies test case dimensionality, 1 or 2 
#plot       (bool) : spefies if you want to make plots or not

parser = argparse.ArgumentParser(description="tmgs_Lmap arguments")

parser.add_argument('--modified', '-m', action='store_true', default=False, help='True/False as in modified or not.')
parser.add_argument('--nstage'  , '-s', type = int,          default=1,     help='Specify nstage.')
parser.add_argument('--verbose' , '-v', action='store_true', default=False, help='Enable verbose output.')
parser.add_argument('--ndim'    , '-d', type = int,          default=1,     help='Specify nstage.')
parser.add_argument('--plot'    , '-p', action='store_true', default=False, help='Enable plotting.')
args = parser.parse_args()

print('modified: ',args.modified)
print('nstage:   ',args.nstage)
print('verbose:  ',args.verbose)
print('ndim:     ',args.ndim)
print('plot:     ',args.plot)

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
    npt  = 50
    x    = rng.uniform(-1,1,(npt))

elif ndim == 2:    # 2d data case

    # function to return 2d tensor-product of monomials phi[k0,k1](x) functions
    # where x is a 2-vector
    def phif_2d(k0,k1):
        def lfnc(x_):
            #return np.array([x[0] * x[1]**k1 for x in x_])
            return np.array([x[0]**k0 * x[1]**k1 for x in x_])
        return lfnc

    nt    = 3
    ntrm  = nt*nt
    phi0  = np.array([phif_2d(k0,k1) for k0 in range(nt) for k1 in range(nt)])

    # define 2d data, an npt x 2 array of npt points, each being a 2-vector
    npt  = 100
    x    = rng.uniform(-1,1,(npt,ndim))

else:
    print('not a valid ndim case option')
    sys.exit(1)

print('ntrm =',ntrm)

#==============================================================================
# specify the linear map, which will also hold the x data

lmap = gso.GLMAP(x, verbose=verbose)                                        # identity map
# lmap = gso.GLMAP(x, 10*np.random.rand(int(npt/2),npt),verbose=verbose)    # user matrix map
# lmap = gso.GLMAP(x, lambda x, u: np.exp(x) * u, verbose=verbose)          # user function map

#==============================================================================
# do mmgs

# build mmgs object
phi  = copy.deepcopy(phi0)
mmgs = gso.MMGS(phi,lmap)

# orthonormalize with multistage modified GS
Pmat, tht = mmgs.ortho(modified=modified,nstage=nstage,verbose=verbose)

print('ortho check phi0 maxabs:',np.max(np.abs(mmgs.ortho_check_phi(0)-np.eye(mmgs.m))))
print('ortho check tht  maxabs:',np.max(np.abs(mmgs.ortho_check(nstage-1)-np.eye(mmgs.m))))

#==============================================================================
# plotting diagnostics
if plot:

    if ndim == 1:   

        nx    = 250
        xt    = np.linspace(-1,1,nx)
        phixt = np.array([f(xt) for f in phi]).T
        thtxt = np.array([f(xt) for f in tht]).T
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) 
        for q in range(mmgs.m):
            ax1.plot(xt, phixt[:,q], label='k:'+str(q))
            ax2.plot(xt, thtxt[:,q], label='k:'+str(q))
        ax1.set_title(r'$\phi$')
        ax2.set_title(r'$\theta$')
        for ax in (ax1, ax2):
            ax.scatter(x,[0]*npt,s=3,label='data')
            ax.set_xlabel(r'$x$')
            ax.grid(True)
        ax1.set_ylabel(r'$\phi_k(x)$')
        ax2.set_ylabel(r'$\theta_k(x)$')
        ax1.set_ylim(min(np.min(phixt),np.min(thtxt)),max(np.max(phixt),np.max(thtxt)))
        ax2.set_ylim(min(np.min(phixt),np.min(thtxt)),max(np.max(phixt),np.max(thtxt)))
        ax2.legend()
        ax2.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        plt.savefig('gso_1d.png')

    elif ndim == 2:

        X    = np.arange(-2, 2, 0.1)
        Y    = np.arange(-1, 1, 0.05)
        nX   = X.shape[0]
        nY   = Y.shape[0]
        X, Y = np.meshgrid(X, Y)
        XY   = np.vstack((X.flatten(),Y.flatten())).T

        Zphi = np.array([phi0[q](XY).reshape(nX,nY) for q in range(mmgs.m)])
        Ztht = np.array([mmgs.mgs[nstage-1].tht[q](XY).reshape(nX,nY) for q in range(mmgs.m)])

        for q in range(mmgs.m):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
            #plt.suptitle('q: '+str(q)+', k0: '+str(int(q/nt))+', k1: '+str(q-int(q/nt)*nt))
            ax1.plot_surface(X, Y, Zphi[q], cmap='viridis')
            ax2.plot_surface(X, Y, Ztht[q], cmap='viridis')
            ax1.set_title(r'$\phi_'+str(q)+'$')
            ax2.set_title(r'$\theta_'+str(q)+'$')
            for ax in (ax1,ax2):
                ax.set_xlabel(r'$x$')
                ax.set_ylabel(r'$y$')
            ax1.set_zlabel(r'$\phi_'+str(q)+'(x,y)$')
            ax2.set_zlabel(r'$\theta_'+str(q)+'(x,y)$')
            ax1.set_zlim(min(np.min(Zphi[q]),np.min(Ztht[q])),max(np.max(Zphi[q]),np.max(Ztht[q])))
            ax2.set_zlim(min(np.min(Zphi[q]),np.min(Ztht[q])),max(np.max(Zphi[q]),np.max(Ztht[q])))
            plt.savefig(f'gso_2d_{q}.png')

    else:
        print('not a valid ndim case option')
        sys.exit(1)
#==============================================================================

print('done')

