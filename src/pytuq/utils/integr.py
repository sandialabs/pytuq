#!/usr/bin/env python
"""Module for integration classes."""

import copy
import numpy as np

from scipy.stats import multivariate_normal, kde
from scipy.integrate import nquad

from .maps import Domainizer
from ..rv.mrv import MCMCRV, GMM


class Integrator():
    """Base class for integration objects."""

    def __init__(self):
        """Initialize."""

    def integrate(self, function, domain=None, func_args=None, xdata=None, **kw_args):
        r"""Dummy method to be implemented by children classes.

        Args:
            function (callable): The function to integrate f(x, **func_args), where the first argument should be a 2d array of size `(N, d)`.
            domain (None, optional): Optionally provide a domain of integration, `(d, 2)`.
            func_args (None, optional): Optional keyword arguments of the function.
            xdata (None, optional): Provide input samples, if needed by the integrator.
            **kw_args: Other keyword arguments necessary for the integratior at hand.

        Raises:
            NotImplementedError: This needs to be implemented by children classes.
        """
        raise NotImplementedError

    def _save(self, a, b, kk):
        """Copying a given key from one dictionary into another.

        Args:
            a (dict): Initial dictionary.
            b (dict): Dictionary to copy to.
            kk (str): The given key.
        """
        if kk in a.keys():
            b[kk] = a[kk].copy()
        else:
            pass


    def integrate_multiple(self, f_arg_list, **kwargs):
        """Integrate multiple functions at once with the same input samples.

        Args:
            f_arg_list (list[tuple]): List of pairs (function, argument).
            **kwargs: Keyword arguments for the function.

        Returns:
            list[float]: List of integral values.

        TODO: assumes all functions take the same keyword arguments.
        """
        integrals = []
        for i, f_args in enumerate(f_arg_list):
            function, args = f_args
            if i>0:
                self._save(results, args, 'saved')
                self._save(results, kwargs, 'xdata')

                #print(i, np.max(args['saved']), np.max(kwargs['xdata']))
            kwargs['func_args']=args
            integral, results = self.integrate(function, **kwargs)
            #print(i, np.max(results['ydata']), np.min(results['ydata']))
            integrals.append(integral)

        return integrals


#  ___    ___  (_)  _ __    _   _
# / __|  / __| | | | '_ \  | | | |
# \__ \ | (__  | | | |_) | | |_| |
# |___/  \___| |_| | .__/   \__, |
#                  |_|      |___/

class IntegratorScipy(Integrator):
    """Integrator that uses SciPy."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def integrate(self, function, domain=None, func_args=None, xdata=None, epsrel=1.e-5):
        """Integration via SciPy.

        Args:
            function (callable): The function to integrate f(x, **func_args), where the first argument should be a 2d array of size `(N, d)`.
            domain (None, optional): Optionally provide a domain of integration, `(d, 2)`.
            func_args (None, optional): Optional keyword arguments of the function.
            xdata (None, optional): Provide input samples, if needed by the integrator.
            epsrel (float, optional): Integration tolerance. Defaults to 1.e-5.

        Returns:
            (float, dict): tuple of integral value and dictionary containing error.
        """
        def wrapper(*args):
            func, dim, func_pars = args[-3:]
            func_input = np.array(args[:dim]).reshape(1, dim)
            return func(func_input, **func_pars)[0]
        assert(xdata is None)
        assert(domain is not None)
        dim, two = domain.shape
        integral, err, results = nquad(wrapper,
                                       domain,
                                       args=(function, dim, func_args),
                                       opts={'epsrel': epsrel},
                                       full_output=True)
        results['err'] = err

        return integral, results


#  _ __ ___     ___   _ __ ___     ___
# | '_ ` _ \   / __| | '_ ` _ \   / __|
# | | | | | | | (__  | | | | | | | (__
# |_| |_| |_|  \___| |_| |_| |_|  \___|

class IntegratorMCMC(Integrator):
    """Integration via MCMC sampling."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def integrate(self, function, domain=None, func_args=None, xdata=None, nmc=100):
        """Integrate via MCMC sampling.

        Args:
            function (callable): The function to integrate f(x, **func_args), where the first argument should be a 2d array of size `(N, d)`.
            domain (None, optional): Optionally provide a domain of integration, `(d, 2)`.
            func_args (None, optional): Optional keyword arguments of the function.
            xdata (None, optional): Provide input samples, if needed by the integrator.
            nmc (int, optional): Number of requested samples. Defaults to 100.

        Returns:
            (float, dict): tuple of integral value and dictionary containing useful info.
        """
        def logfunction(x, **args):
            assert(len(x.shape) == 1)
            return np.log(function(x.reshape(1, -1), **args)[0])
        assert(domain is not None)
        dim, two = domain.shape

        if xdata is None:
            rv = MCMCRV(dim, logfunction, param_ini=np.ones((dim,)), nmcmc=10 * nmc)
            xdata = rv.sample(nmc, **func_args)
        ydata = function(xdata, **func_args)

        kde_py = kde.gaussian_kde(xdata.T)
        kde_weight = kde_py(xdata.T)

        integral = np.mean(ydata / kde_weight)

        results = {'err': None, 'neval': nmc, 'xdata': xdata, 'ydata': ydata, 'icw': kde_weight}
        return integral, results


#  _ __ ___     ___
# | '_ ` _ \   / __|
# | | | | | | | (__
# |_| |_| |_|  \___|

class IntegratorMC(Integrator):
    """Integrator via Monte-Carlo sampling."""

    def __init__(self, seed=None):
        """Initialize."""
        super().__init__()
        np.random.seed(seed=seed)

    def integrate(self, function, domain=None, func_args=None, xdata=None, nmc=100):
        """Integration via Monte-Carlo.

         Args:
            function (callable): The function to integrate f(x, **func_args), where the first argument should be a 2d array of size `(N, d)`.
            domain (None, optional): Optionally provide a domain of integration, `(d, 2)`.
            func_args (None, optional): Optional keyword arguments of the function.
            xdata (None, optional): Provide input samples, if needed by the integrator.
            nmc (int, optional): Number of requested samples. Defaults to 100.

        Returns:
            (float, dict): tuple of integral value and dictionary containing useful info.
        """
        assert(domain is not None)
        dim, two = domain.shape

        if xdata is None:
            sc = Domainizer(domain)
            xdata = sc.inv(np.random.rand(nmc, dim))
        ydata = function(xdata, **func_args)

        volume = np.prod(domain[:, 1] - domain[:, 0])
        integral = volume * np.mean(ydata)
        results = {'err': None, 'neval': nmc, 'xdata': xdata, 'ydata': ydata}
        return integral, results

# __      __  _ __ ___     ___
# \ \ /\ / / | '_ ` _ \   / __|
#  \ V  V /  | | | | | | | (__
#   \_/\_/   |_| |_| |_|  \___|

class IntegratorWMC(Integrator):
    """Integrator via weighted Monte-Carlo."""

    def __init__(self, seed=None):
        """Initialize."""
        super().__init__()
        np.random.seed(seed=seed)

    def integrate(self, function, func_args=None, xdata=None, mean=None, cov=None,
                  nmc=100):
        """Integration by weighted single Gaussian.

        Args:
            function (callable): The function to integrate f(x, **func_args), where the first argument should be a 2d array of size `(N, d)`.
            func_args (None, optional): Optional keyword arguments of the function.
            xdata (None, optional): Provide input samples, if needed by the integrator.
            mean (None, optional): Mean array of size `d` of the associated gaussian weight. This argument is not optional.
            cov (None, optional): Covariance matrix of size `(d,d)` of the associated gaussian weight. Defaults to identity matrix.
            nmc (int, optional): Number of requested samples. Defaults to 100.

        Returns:
            (float, dict): tuple of integral value and dictionary containing useful info.
        """
        assert(mean is not None)
        if cov is None:
            cov = np.eye(mean.shape[0])

        if xdata is None:
            xdata = np.random.multivariate_normal(mean,
                                              cov,
                                              size=(nmc, ))
        ydata = function(xdata, **func_args)

        gaussian_weight = multivariate_normal.pdf(xdata,
                                                  mean=mean,
                                                  cov=cov)

        integral = np.mean(ydata / gaussian_weight)

        results = {'err': None, 'neval': nmc, 'xdata': xdata, 'ydata': ydata, 'icw': gaussian_weight}
        return integral, results

#   __ _   _ __ ___    _ __ ___
#  / _` | | '_ ` _ \  | '_ ` _ \
# | (_| | | | | | | | | | | | | |
#  \__, | |_| |_| |_| |_| |_| |_|
#  |___/

class IntegratorGMM(Integrator):
    """Integrator via Gaussian mixture models."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def integrate(self, function,
                  func_args=None, xdata=None,
                  weights=None, means=None, covs=None,
                  nmc=100):
        """Integration by Gaussian mixtures.

        Args:
            function (callable): The function to integrate f(x, **func_args), where the first argument should be a 2d array of size `(N, d)`.
            func_args (None, optional): Optional keyword arguments of the function.
            xdata (None, optional): Provide input samples, if needed by the integrator.
            weights (list[float], optional): Weights of associated mixtures. Defaults to None, which automatically picks the weights based on function values and its width at the peaks.
            means (list[np.ndarray], optional): List of mean arrays. This is not optional.
            covs (list[np.ndarray], optional): List of covariance matrices.
            nmc (int, optional): Number of requested samples. Defaults to 100.

        Returns:
            (float, dict): tuple of integral value and dictionary containing useful info.
        """
        assert(means is not None)
        if weights is None:
            weights = [function(mean.reshape(1,-1), **func_args)[0] * np.sqrt(np.linalg.det(cov)) for mean, cov in zip(means, covs)]

        mygmm = GMM(means, covs=covs, weights=weights)

        if xdata is None:
            xdata = mygmm.sample(nmc)
        ydata = function(xdata, **func_args)

        gmm_pdf = mygmm.pdf(xdata)

        integral = np.mean(ydata / gmm_pdf)

        results = {'err': None, 'neval': nmc, 'xdata': xdata, 'ydata': ydata, 'icw': gmm_pdf}
        return integral, results

#                                  _
#   __ _   _ __ ___    _ __ ___   | |_
#  / _` | | '_ ` _ \  | '_ ` _ \  | __|
# | (_| | | | | | | | | | | | | | | |_
#  \__, | |_| |_| |_| |_| |_| |_|  \__|
#  |___/

class IntegratorGMMT(Integrator):
    """Integrator via truncated Gaussian mixture models."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def integrate(self, function, domain=None,
                  func_args=None, xdata=None, weights=None, means=None, covs=None,
                  nmc=100):
        """Integration by truncated Gaussian mixtures.

        Args:
            function (callable): The function to integrate f(x, **func_args), where the first argument should be a 2d array of size `(N, d)`.
            domain (None, optional): Provide a domain of integration/truncation, `(d, 2)`.
            func_args (None, optional): Optional keyword arguments of the function.
            xdata (None, optional): Provide input samples, if needed by the integrator.
            weights (list[float], optional): Weights of associated mixtures. Defaults to None, which automatically picks the weights based on function values and its width at the peaks.
            means (list[np.ndarray], optional): List of mean arrays. This is not optional.
            covs (list[np.ndarray], optional): List of covariance matrices.
            nmc (int, optional): Number of requested samples. Defaults to 100.

        Returns:
            (float, dict): tuple of integral value and dictionary containing useful info.
        """

        assert(means is not None)
        if weights is None:
            # TODO: this won't work well if wrapped in integrate_multiple()
            weights = [function(mean.reshape(1,-1), **func_args)[0] * np.sqrt(np.linalg.det(cov)) for mean, cov in zip(means, covs)]

        #print(weights)
        mygmm = GMM(means, covs=covs, weights=weights)
        #print(mygmm.weights)

        if xdata is None:
            xdata = mygmm.sample_indomain(nmc, domain)
        ydata = function(xdata, **func_args)

        gmm_pdf = mygmm.pdf(xdata)
        volume = mygmm.volume_indomain(domain)
        integral = volume * np.mean(ydata / gmm_pdf)
        #print("AAA ", gmm_pdf)

        results = {'err': None, 'neval': nmc, 'xdata': xdata, 'ydata': ydata, 'icw': gmm_pdf/volume}
        return integral, results


