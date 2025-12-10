#!/usr/bin/env python
"""Module for KL+Surrogate construction."""

import os
import sys
import numpy as np

from .kle import KLE

from ..gsa.gsa import SamSobol
from ..workflows.fits import pc_fit
from ..func.func import Function
from ..utils.plotting import myrc, set_colors, plot_sens, plot_dm
from ..utils.mindex import micf_join
from ..rv.pcrv import PCRV

try:
    from quinn.nns.mlp import MLP
    from quinn.nns.rnet import RNet, NonPar
except ImportError:
    print("Warning: QUiNN not installed. NN functionality won't work.")
    sys.exit()


myrc()

# This should replace KLNN, and elm_glob should use this instead of KLNN?
class KLSurr():
    r"""KL+Surrogate Class.

    Attributes:
        built (bool): Indicates whether KLSurr is built or not.
        function (callable): A callable function that is the KLSurr evaluator.
        kl (KLE): The underlying KLE object.
        ndim (int): The input dimensionality.
        neig (int): The number of retained eigenvalues in the KLE.
        smodel (torch.nn or PCRV): Surrogate model object, either NN or PC.
        xisurr (callable): Surrogate evaluator of the latent features.
        xitrain_kl (np.ndarray): KL latent features.
        ytrain_kl (np.ndarray): KL-compressed training data.
        ytrain_klsurr (np.ndarray): KL+Surrogate approximation of the training data.
    """

    def __init__(self):
        """Initialization."""
        self.ndim = None
        self.built = False
        self.xisurr = None
        self.smodel = None
        self.function = None
        self.neig = None

    def setDim(self, ndim):
        r"""Set the input dimensionality.

        Args:
            ndim (int): Input dimensionality, `d`.
        """
        self.ndim = ndim

    def build(self, ptrain, ytrain, neig=None, surr='PC', tr_frac=0.9, pc_order=3, bcs_eta=1.e-8):
        """Build the KL+Surrogate.

        Args:
            ptrain (np.ndarray): An input array of size `(M,d)`.
            ytrain (np.ndarray): Output training data array of size `(M,N)`.
            neig (int, optional): Number of eigenvalues. Defaults to None, which means all eigenvalues are retained.
            surr (str, optional): Surrogate type, PC or NN. Defaults to PC.
            tr_frac (float, optional): Fraction of samples used for training. Defaults to 0.9.
            pc_order (int, optional): PC order. Defaults to 3.
            bcs_eta (float, optional): BCS tolerance. Defaults to 1.e-8.
        """
        nens, ndim = ptrain.shape
        nens_, nt = ytrain.shape
        assert(nens==nens_)


        self.setDim(ndim=ndim)

        # Build KL expansion
        self.kl = KLE()
        self.kl.build(ytrain.T, plot=True)

        # Pick the first neig eigenvalues that explain 99% of variance
        if neig is None:
            neig = self.kl.get_neig(0.99)


        ######################################################################
        # Construct surrogates for KL modes
        ######################################################################

        xitrain = self.kl.xi[:, :neig]
        nout = xitrain.shape[1] # number of outputs to build surrogates for is neig

        if surr == 'NN':
            # mynn = MLP(ndim, nout, (100, 200, 200))
            mynn = RNet(33, 3, NonPar(4),
                indim=ndim, outdim=nout, layer_pre=True, layer_post=True, biasorno=True, nonlin=True, mlp=False)

            indperm = np.random.permutation(range(nens))
            ntrn = int(tr_frac*nens)
            nval = nens - ntrn
            indtrn = indperm[:ntrn]
            indval = indperm[-nval:]
            if ntrn == nens:
                indval = indtrn

            nnmodel = mynn.fit(ptrain[indtrn], xitrain[indtrn],
                         val=[ptrain[indval], xitrain[indval]],
                         lrate=0.001, batch_size=10, nepochs=2000,
                         gradcheck=False, freq_out=100)
            self.xisurr = lambda x: mynn.predict(x)
            self.smodel = nnmodel

        elif surr == 'PC':
            pcrv, _ = pc_fit(ptrain, xitrain,
                             order=pc_order, pctype='LU', method='bcs', eta = bcs_eta)
            self.xisurr = lambda x: pcrv.function(x)
            self.smodel = pcrv

        else:
            print(f"Surrogate type {surr} is unknown. Please use NN or PC. Exiting.")
            sys.exit()


        self.built = True



        self.ytrain_kl = self.kl.eval(neig=neig)
        self.xitrain_kl = self.evalxi(ptrain)
        self.ytrain_klsurr = self.kl.eval(xi=self.xitrain_kl, neig=neig)
        self.neig = neig

        self.setFunction()

    def setFunction(self):
        """Creates the function evaluator corresponding to the surrogate."""
        self.function = Function(name='KLSurr Function')
        self.function.setDimDom(dimension=self.ndim)
        self.function.setCall(lambda x: self.eval(x).T)

    def evalxi(self, ptrain, otherpars=None):
        r"""Evaluates new latent features at given inputs.

        Args:
            ptrain (np.ndarray): Input array of size `(M, d)`.
            otherpars (None, optional): Optional parameters (unused currently).

        Returns:
            np.ndarray: Predictions of latent features, an array of size `(M, K)`.
        """
        assert(self.built)
        xi_pred = self.xisurr(ptrain)

        return xi_pred

    def eval(self, ptrain, neig=None):
        r"""Evaluate the KL+Surrogate at given inputs.

        Args:
            ptrain (np.ndarray): Input array of size `(M, d)`.
            neig (int, optional): Number of eigenvalues. Defaults to None, which means all eigenvalues are retained.

        Returns:
            np.ndarray: Output array of size `(M,N)`.
        """
        assert(self.built)


        xi = self.evalxi(ptrain)
        if neig is None:
            neig = xi.shape[1]
        ytrain_klsurr = self.kl.eval(xi=xi, neig=neig)

        return ytrain_klsurr


    def plot_parity_xi(self):
        """Plots diagonal parity plots for each latent feature."""
        neig = self.xitrain_kl.shape[1]
        for i in range(neig):
            plot_dm([self.kl.xi[:, i]], [self.xitrain_kl[:,i]],
                    axes_labels=[r'$\xi_{'+f'{i+1}'+'}$', r'$\xi_{'+f'{i+1}'+'}^'+'{surr}$'],
                    labels=['Training'], msize=7,
                    figname=f'xi_dm_{str(i).zfill(3)}.png')
            # fig = plt.figure(figsize=(12,10))
            # plt.plot(self.kl.xi[:, i], self.xitrain_kl[:,i], 'bo')
            # plt.xlabel('ELM')
            # plt.ylabel('Apprx.')

            # plt.savefig(f'xi_dm_{i}.png')
            # plt.clf()

    # TODO: this need cleanup, and perhaps there is no need for this here, as long as klsurr.function is set!
    # TODO: plotting can really be done outside
    def compute_sens(self, domain, npar, colors=None, pnames=None, totmain='main'):
        r"""Computes and plots Sobol sensitivities via sampling.

        Args:
            domain (np.ndarray): Domain of input, array of size `(d,2)`.
            npar (int): Number of parameters.
            colors (None, optional): Optional list of colors for plotting.
            pnames (None, optional): Optional list of names for plotting.
            totmain (str, optional): Total or main sensitivity. Defaults to main.

        Returns:
            np.ndarray: Returns Sobol sensitivities, an array of size `(N,d)`.
        """
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
            allsens_eig_sobol[i, :] = sens[totmain]

        if colors is None:
            colors = set_colors(ndim)
        if pnames is None:
            pnames = [f'p{j}' for j in range(1, ndim+1)]

        plot_sens(allsens_eig_sobol,range(ndim),range(neig),vis="bar",reverse=False,
                     par_labels=pnames, case_labels=[r'$\xi_{'+str(j+1)+'}$' for j in range(neig)],
                     colors=colors,ncol=5,grid_show=False,
                     xlbl='Eigen-features',legend_show=3,legend_size=25,xdatatick=None,
                     figname='sens_eig.png', topsens=None, lbl_size=30, yoffset=0.01,
                     title='', xticklabel_size=20, xticklabel_rotation=0)

        # Sensitivities for the actual model
        ysam_model = self.eval(qsam, neig=neig).T # same as klnn.kl.eval(xi=ysam_eig, neig=neig).T
        nx = ysam_model.shape[1]
        allsens_sobol = np.empty((nx, ndim))
        for i in range(nx):
            sens = SensMethod.compute(ysam_model[:, i])
            allsens_sobol[i, :] = sens[totmain]

        return allsens_sobol

    # TODO: plotting can really be done outside
    def compute_sens_pc(self, colors=None, pnames=None, totmain='main'):
        r"""Computes PC-based sensitivities if PC surrogate is used.

        Args:
            colors (None, optional): Optional list of colors for plotting.
            pnames (None, optional): Optional list of names for plotting.
            totmain (str, optional): Total or main sensitivity. Defaults to main.

        Returns:
            np.ndarray: Returns Sobol sensitivities, an array of size `(N,d)`.
        """
        assert(isinstance(self.smodel, PCRV))
        mindex_all, cfs_all = micf_join(self.smodel.mindices, self.smodel.coefs)
        assert(np.sum(mindex_all[0, :]) == 0)
        assert(self.neig==cfs_all.shape[0])
        assert(len(set(self.smodel.pctypes))==1) #assert all PCs are of the same type

        # Sensitivities for eigenmodes
        pcrv_eig=PCRV(self.neig, self.ndim, self.smodel.pctypes, mi=mindex_all, cfs=cfs_all)

        if totmain=='main':
            allsens_eig_sobol = pcrv_eig.computeSens()
        elif totmain=='total':
            allsens_eig_sobol = pcrv_eig.computeTotSens()
        else:
            print("Please indicate main or total sensititvity. Exiting.")
            sys.exit()

        if colors is None:
            colors = set_colors(self.ndim)
        if pnames is None:
            pnames = [f'p{j}' for j in range(1, self.ndim+1)]

        plot_sens(allsens_eig_sobol,range(self.ndim),range(self.neig),vis="bar",reverse=False,
                     par_labels=pnames, case_labels=[r'$\xi_{'+str(j+1)+'}$' for j in range(self.neig)],
                     colors=colors,ncol=5,grid_show=False,
                     xlbl='Eigen-features',legend_show=3,legend_size=25,xdatatick=None,
                     figname='sens_eig.png', topsens=None, lbl_size=30, yoffset=0.01,
                     title='', xticklabel_size=20, xticklabel_rotation=0)

        # Sensitivities for the actual model
        cfs_glob = np.dot(np.dot(cfs_all.T, np.diag(np.sqrt(self.kl.eigval[:self.neig]))), \
            self.kl.modes[:, :self.neig].T) #npc, nx

        cfs_glob[0, :] += self.kl.mean


        print("Multiindex shape :", mindex_all.shape)
        print("PC coefficients' shape :", cfs_glob.shape)#npc, nx
        nx = cfs_glob.shape[1]
        pcrv_phys=PCRV(nx, self.ndim, self.smodel.pctypes, mi=mindex_all, cfs=cfs_glob.T)
        if totmain=='main':
            allsens_sobol = pcrv_phys.computeSens()
        elif totmain=='total':
            allsens_sobol = pcrv_phys.computeTotSens()
        else:
            print("Please indicate main or total sensititvity. Exiting.")
            sys.exit()

        return allsens_sobol

