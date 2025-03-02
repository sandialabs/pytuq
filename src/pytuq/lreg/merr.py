#!/usr/bin/env python
"""Module for linear regression with embedded model error."""

import sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.linalg import lstsq


from .lreg import lreg
from ..minf.mcmc import AMCMC


def map_flat_to_tri(ij, n, embedding):
    r"""Mapping a flattened index of embedded parameters into a pair of indices.

    Args:
        ij (int): Flattened parameter index.
        n (int): Number of physical parameters.
        embedding (string): Embedding type.

    Returns:
        tuple(int, int): A tuple of indices :math:`(i,j)`, to fill a triangular matrix of lower-Cholesky factor of the covariance.
    """
    assert(n>0)
    assert(ij>=0)
    if embedding=='iid':
        i, j = ij, ij
    elif embedding=='mvn':
        assert(ij<n*(n+1)/2)
        counter = 0
        k = -1
        while counter <= ij:
            k+=1
            counter += (k+1)
        i = k
        j = ij - i*(i+1)//2

    else:
        print(f"Embedding {embedding} unknown. Exiting.")
        sys.exit()
    #print(i,j)
    assert(i<n and j<n)
    return i, j



def logpost_emb(x, aw=None, bw=None, ind_embed=None, dvinfo=None, multiplicative=False, merr_method='abc', embedding='iid', mean_fixed=False, cfs_mean=None):
    """Log-posterior of embedded model.

    Args:
        x (np.ndarray): An 1d array input to log-posterior, i.e. the dimensionality is the chain dimensionality.
        aw (None, optional): The A-matrix of size :math:`(N,K)` of basis evaluations at training points. Default None should not be used, and is made keyword out of convenience.
        bw (None, optional): An 1d array of data if size :math:`N`. Default None should not be used, and is made keyword out of convenience.
        ind_embed (None, optional): Indices of parameters/coefficients/bases to embed model error in. Defaults to None, which means embed in all.
        dvinfo (dict, optional): Dictionary of parameters controlling data variance. Keys are 'npar' (number of parameters), 'fcn' (callable to return data variance vector given the last part of the chain, and auxiliary paraemters), 'aux' (auxiliary parameters of the data variance function). Default is None, but should not be used.
        multiplicative (bool, optional): Indicates whether the embedding is multiplicative. Defaults to False.
        merr_method (str, optional): Model error embedding method. Options are 'full', 'iid', 'abc' (default).
        embedding (str, optional): Model error embedding type. Options are 'iid' (default) and 'mvn'.
        mean_fixed (bool, optional): Whether the mean fit parameters are fixed or it is being inferred together with the model error parameters. Defaults to False.
        cfs_mean (None, optional): Mean coefficient vector. Part (or the whole, depending on ind_embed) of this will be overwritten if mean_fixed is False. Default is None, but it should never be used.

    Returns:
        float: Log-posterior value.
    """
    assert(aw is not None and bw is not None)
    assert(isinstance(dvinfo, dict))
    npt, nbas = aw.shape
    nchain = x.shape[0]

    if ind_embed is None:
        ind_embed = range(nbas)
    nbas_emb = len(ind_embed)

    # Set data variance
    dvnpar = dvinfo['npar']
    data_variance = dvinfo['fcn'](x[nchain-dvnpar:], dvinfo['aux'])

    assert(cfs_mean is not None)
    cfs = cfs_mean.copy()
    if mean_fixed:
        coefs_flat = x[:nchain-dvnpar]
    else:
        cfs[ind_embed] = x[:nbas_emb]
        coefs_flat = x[nbas_emb:nchain-dvnpar]

    # if(np.min(coefs_flat)<=0.0):
    #     return -1.e+80


    coefs_Lmat = np.zeros((nbas,nbas))
    for ij in range(coefs_flat.shape[0]):
        i, j = map_flat_to_tri(ij, nbas_emb, embedding)
        coefs_Lmat[ind_embed[i], ind_embed[j]] = coefs_flat[ij]


    if multiplicative:
        coefs_Lmat = np.dot(np.diag(np.abs(cfs)),coefs_Lmat)


    ss = aw @ coefs_Lmat

    # #### FULL COVARIANCE
    if merr_method == 'full':
        cov = ss @ ss.T + np.diag(data_variance) * np.eye(npt)
        val = multivariate_normal.logpdf(aw @ cfs, mean=bw, cov=cov, allow_singular=True)

    # #### IID
    elif merr_method == 'iid':
        err = aw @ cfs - bw
        stds = np.linalg.norm(ss, axis=1)
        stds = np.sqrt(stds**2+data_variance)
        val = -0.5 * np.sum((err/stds)**2)
        val -= 0.5 * npt * np.log(2.*np.pi)
        val -= np.sum(np.log(stds))

    #### ABC
    elif merr_method == 'abc':
        abceps=0.01
        abcalpha=1.0
        err = aw @ cfs - bw
        stds = np.linalg.norm(ss, axis=1)
        stds = np.sqrt(stds**2+data_variance)
        err2 = abcalpha*np.abs(err)-stds
        val = -0.5 * np.sum((err/abceps)**2)
        val -= 0.5 * np.sum((err2/abceps)**2)
        val -= 0.5 * np.log(2.*np.pi)
        val -= np.log(abceps)

    else:
        print(f"Merr type {merr_method} unknown. Exiting.")
        sys.exit()

    # Prior?
    #val -= np.sum(np.log(np.abs(sig_cfs)))

    return val



class lreg_merr(lreg):
    r"""A class for model error embedded linear regression.

    Attributes:
        cf (np.ndarray): An 1d array of coefficients, of size :math:`K`.
        cf_cov (np.ndarray): A 2d array of coefficient covariance of size :math:`(K,K)`.
        dvinfo (dict): Dictionary of parameters controlling data variance. Keys are 'npar' (number of parameters), 'fcn' (callable to return data variance vector given the last part of the chain, and auxiliary paraemters), 'aux' (auxiliary parameters of the data variance function).
        embedding (str): Model error embedding type. Options are 'iid' and 'mvn'.
        fitted (bool): Indicates whether the fit is already performed or not.
        ind_embed (np.ndarray): Indices of parameters/coefficients/bases to embed model error in.
        mean_fixed (bool): Whether the mean fit parameters are fixed or it is being inferred together with the model error parameters.
        merr_method (str): Model error embedding method. Options are 'full', 'iid', 'abc'.
        multiplicative (bool): Indicates whether the embedding is multiplicative.
        opt_method (str): Optimization methods. Options are 'bfgs' or 'mcmc'.
    """

    def __init__(self, ind_embed=None, dvinfo=None, multiplicative=False, merr_method='abc', opt_method='bfgs', mean_fixed=False, embedding='iid'):
        """Initialization.

        Args:
            dvinfo (None, optional): Description
            multiplicative (bool, optional): Description
            merr_method (str, optional): Description
            opt_method (str, optional): Description
            mean_fixed (bool, optional): Description
            embedding (str, optional): Description

            ind_embed (None, optional): Indices of parameters/coefficients/bases to embed model error in. Defaults to None, which means embed in all.
            dvinfo (dict, optional): Dictionary of parameters controlling data variance. Keys are 'npar' (number of parameters), 'fcn' (callable to return data variance vector given the last part of the chain, and auxiliary paraemters), 'aux' (auxiliary parameters of the data variance function). Default is None, but should not be used.
            multiplicative (bool, optional): Indicates whether the embedding is multiplicative. Defaults to False.
            merr_method (str, optional): Model error embedding method. Options are 'full', 'iid', 'abc' (default).
            opt_method (str, optional): Optimization methods. Options are 'bfgs' (default) or 'mcmc'.
            mean_fixed (bool, optional): Whether the mean fit parameters are fixed or it is being inferred together with the model error parameters. Defaults to False.
            embedding (str, optional): Model error embedding type. Options are 'iid' (default) and 'mvn'.
        """
        super().__init__()

        self.ind_embed = ind_embed
        if dvinfo is None:
            self.dvinfo = {'fcn': lambda v, p: 0.0, 'npar': 0, 'aux': None}
        else:
            self.dvinfo = dvinfo
        self.multiplicative = multiplicative
        self.merr_method = merr_method
        self.opt_method = opt_method
        self.mean_fixed = mean_fixed
        self.embedding = embedding

    def fita(self, A, y):
        r"""Fit given A-matrix of basis evaluations and data array.

        Args:
            A (np.ndarray): A 2d array of size :math:`(N, K)` each row holding basis evaluations at a training point.
            y (np.ndarray): An 1d array of size :math:`N` holding the data.
        """
        npts, nbas = A.shape
        assert(y.shape[0] == npts)

        if self.ind_embed is None:
            self.ind_embed = range(nbas)

        nbas_emb = len(self.ind_embed)
        if self.embedding == 'mvn':
            num_merr_embed = int(nbas_emb*(nbas_emb+1)/2)
        elif self.embedding == 'iid':
            num_merr_embed = nbas_emb+0
        else:
            print(f"Embedding type {self.embedding} is unknown. Exiting.")
            sys.exit()


        #cfs_mean, residues, rank, s = lstsq(A, y, 1.0e-13)
        invptp = np.linalg.inv(np.dot(A.T, A)+1.e-6*np.diag(np.ones((nbas,)))) # TODO: this nugget is dangerous if overall scales are small
        cfs_mean = np.dot(invptp, np.dot(A.T, y))
        #cfs_mean -= np.random.rand(cfs_mean.shape[0])


        logpost_params = {'aw': A, 'bw':y, 'ind_embed':self.ind_embed, 'dvinfo':self.dvinfo, 'multiplicative':self.multiplicative, 'merr_method':self.merr_method, 'embedding':self.embedding, 'mean_fixed':self.mean_fixed, 'cfs_mean':cfs_mean}


        if self.mean_fixed:
            params_ini = np.random.rand(num_merr_embed+self.dvinfo['npar'])
        else:
            params_ini = np.random.rand(nbas_emb+num_merr_embed+self.dvinfo['npar'])
            params_ini[:nbas_emb] = cfs_mean[self.ind_embed]


        nchain = params_ini.shape[0]

        if self.opt_method == 'mcmc':

            # res = minimize((lambda x, fcn, p: -fcn(x, **p)), params_ini, args=(logpost_emb,logpost_params), method='BFGS', options={'gtol': 1e-16})
            # print(res)
            # params_ini = res.x

            covini = 0.1 * np.ones((nchain, nchain))
            nmcmc = 10000
            gamma = 0.05
            t0 = 100
            tadapt = 100
            calib_params = {'cov_ini': covini,
                            't0': t0, 'tadapt' : tadapt,
                            'gamma' : gamma}
            calib = AMCMC(**calib_params)
            #samples, cmode, pmode, acc_rate, acc_rate_all, pmode_all = amcmc([nmcmc, params_ini, gamma, t0, tadapt, covini], logpost_emb, A, y, ind_sig=ind_embed, sgn=1.0)

            calib.setLogPost(logpost_emb, None, **logpost_params)
            calib_results = calib.run(nmcmc, params_ini)

            samples, cmode, pmode, acc_rate = calib_results['chain'],  calib_results['mapparams'],calib_results['maxpost'], calib_results['accrate']

            np.savetxt('chn.txt', samples)
            np.savetxt('mapparam.txt', cmode)

            solution = cmode

        elif self.opt_method == 'bfgs':
            #params_ini[nbas:] = np.random.rand(nbas_emb,)
            res = minimize((lambda x, fcn, p: -fcn(x, **p)), params_ini, args=(logpost_emb, logpost_params), method='BFGS', options={'gtol': 1e-3})
            print(res)
            solution = res.x

        else:
            print(f"Method {self.opt_method} is unknown. Exiting.")
            sys.exit()

        self.cf = cfs_mean.copy()
        if self.mean_fixed:
            coefs_flat = solution[:nchain-self.dvinfo['npar']]
        else:
            self.cf[self.ind_embed] = solution[:nbas_emb]
            coefs_flat = solution[nbas_emb:nchain-self.dvinfo['npar']]

        # TODO: unused so far
        varpar = solution[nchain-self.dvinfo['npar']:]

        coefs_Lmat = np.zeros((nbas,nbas))
        for ij in range(coefs_flat.shape[0]):
            i, j = map_flat_to_tri(ij, nbas_emb, self.embedding)
            #print(ij, i, j)
            coefs_Lmat[self.ind_embed[i], self.ind_embed[j]] = coefs_flat[ij]

        #print(solution, nbas_emb, nchain, self.dvinfo['npar'], coefs_flat)

        if self.multiplicative:
            coefs_Lmat = np.dot(np.diag(np.abs(self.cf)),coefs_Lmat)


        self.cf_cov = np.dot(coefs_Lmat, coefs_Lmat.T)
        self.fitted = True

        return
