#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

from .kle import KLE

from ..gsa.gsa import SamSobol
from ..utils.plotting import myrc, set_colors, plot_sens, plot_dm

try:
    from quinn.nns.mlp import MLP
except ImportError:
    print("Warning: QUiNN not installed. NN functionality won't work.")

myrc()

class KLNN():
    """docstring for KLNN"""
    def __init__(self):
        return

    def build(self, ptrain, ytrain, neig=None, tr_frac=0.9):

        nens, ndim = ptrain.shape
        nens_, nt = ytrain.shape
        assert(nens==nens_)
        assert(nt>1) # no need to do this for 1 condition

        # Build KL expansion
        self.kl = KLE()
        self.kl.build(ytrain.T, plot=True)

        # Pick the first neig eigenvalues that explain 99% of variance
        if neig is None:
            explained_variance = np.cumsum(self.kl.eigval)/np.sum(self.kl.eigval)
            neig = np.where(explained_variance>0.99)[0][0]+1
            #neig = max(neig, 2) # pick at least two, otherwise array size errors - too lazy to fix now
        print(f'Number of eigenvalues requested : {neig}')



        ######################################################################
        # Construct surrogates for KL modes
        ######################################################################

        xitrain = self.kl.xi[:, :neig]
        nout = xitrain.shape[1] # number of outputs to build surrogates for is neig

        self.mynn = MLP(ndim, nout, (20, 20, 20))

        indperm = np.random.permutation(range(nens))
        ntrn = int(tr_frac*nens)
        ntst = nens - ntrn
        indtrn = indperm[:ntrn]
        indtst = indperm[-ntst:]
        if ntrn == nens:
            indtst = indtrn

        nnmodel = self.mynn.fit(ptrain[indtrn, :], xitrain[indtrn, :],
                     val=[ptrain[indtst, :], xitrain[indtst, :]],
                     lrate=0.001, batch_size=20, nepochs=2000,
                     gradcheck=False, freq_out=100)

        self.built = True



        self.ytrain_kl = self.kl.eval(neig=neig)
        self.xitrain_kl = self.evalxi(ptrain)
        self.ytrain_klnn = self.kl.eval(xi=self.xitrain_kl, neig=neig)
        self.neig = neig


    def evalxi(self, ptrain, otherpars=None):
        assert(self.built)
        xi_pred = self.mynn.predict(ptrain, trained=True)

        return xi_pred

    def eval(self, ptrain, neig=None):
        assert(self.built)


        xi = self.evalxi(ptrain)
        if neig is None:
            neig = xi.shape[1]
        ytrain_klnn = self.kl.eval(xi=xi, neig=neig)

        return ytrain_klnn

    def plot_parity_xi(self):
        neig = self.xitrain_kl.shape[1]
        for i in range(neig):
            plot_dm([self.kl.xi[:, i]], [self.xitrain_kl[:,i]],
                    axes_labels=[r'$\xi_{'+f'{i+1}'+'}$', r'$\xi_{'+f'{i+1}'+'}^'+'{surr}$'],
                    labels=['Training'], msize=7,
                    figname=f'xi_dm_{str(i).zfill(3)}.png')

    def compute_sens(self, domain, npar, colors=None, pnames=None):
        ndim, two = domain.shape
        assert(two==2)
        SensMethod = SamSobol(domain)
        qsam = SensMethod.sample(npar)
        ysam_eig = self.evalxi(qsam)
        neig = ysam_eig.shape[1]

        # Sensitivities for eigenmodes
        allsens_eig_sobol = np.empty((neig, ndim))
        for i in range(neig):
            sens = SensMethod.compute(ysam_eig[:, i])
            allsens_eig_sobol[i, :] = sens['total'] # or 'main'

        if colors is None:
            colors = set_colors(ndim)
        if pnames is None:
            pnames = [f'p{j}' for j in range(1, dim+1)]

        plot_sens(allsens_eig_sobol,range(ndim),range(neig),vis="bar",reverse=False,
                     par_labels=pnames, case_labels=[r'$\xi_{'+str(j+1)+'}$' for j in range(neig)],
                     colors=colors,ncol=5,grid_show=False,
                     xlbl='Eigen-features',legend_show=3,legend_size=25,xdatatick=[],
                     figname='sens_eig.png',showplot=False,topsens=None, lbl_size=30, yoffset=0.01,
                     title='', xticklabel_size=20, xticklabel_rotation=0)

        # Sensitivities for the actual model
        ysam_model = self.eval(qsam, neig=neig).T # same as klnn.kl.eval(xi=ysam_eig, neig=neig).T
        nx = ysam_model.shape[1]
        allsens_sobol = np.empty((nx, ndim))
        for i in range(nx):
            sens = SensMethod.compute(ysam_model[:, i])
            allsens_sobol[i, :] = sens['total'] # or 'main'

        return allsens_sobol



        return
