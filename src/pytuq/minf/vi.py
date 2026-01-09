#!/usr/bin/env python
"""Variational Inference module.
"""

import numpy as np
import matplotlib.pyplot as plt

from pytuq.rv.mrv import Normal_1d

class MFVI():
    def __init__(self, model, y_data, pdim, lossinfo, reparam=None, priors=None):
        """Mean-Field Variational Inference class

        Args:
            model (callable): Forward model.
            y_data (np.ndarray): Data array.
            pdim (int): Parameter dimensionality.
            lossinfo (dict): Dictionary of loss computation auxiliary parameters. Keys are 'nmc' and 'datasigma'.
            reparam (str, optional): Input parameter sigma reparameterization. Defaults to no reparameterization.
            priors (list[pytuq.rv.mrv.MRV], optional): Defaults to None, which is i.i.d. normal with wide variance.

        """
        self.model = model
        self.pdim = pdim
        self.y_data = y_data

        self.reparam = reparam
        if self.reparam == 'exp':
            self.sigma = self.sigma_exp
        elif self.reparam == 'logexp':
            self.sigma = self.sigma_logexp
        elif self.reparam == 'idt' or self.reparam is None:
            self.sigma = self.sigma_idt
        else:
            raise ValueError(f"Unknown reparameterization method: {self.reparam}")

        self.priors = priors
        if self.priors is None:
            self.priors = [Normal_1d(0, 100)] * self.pdim

        self.nmc = lossinfo['nmc']
        if isinstance(lossinfo['datasigma'], (float, int)):
            self.datasigma = lossinfo['datasigma'] * np.ones_like(self.y_data)
        else:
            self.datasigma = lossinfo['datasigma']

    def sigma_exp(self, rho):
        """Exponential reparameterization for standard deviation.

        Args:
            rho (float): Input parameter.

        Returns:
            float: Standard deviation."""
        return np.exp(rho)

    def sigma_logexp(self, rho):
        """Log1-Exponential reparameterization for standard deviation.

        Args:
            rho (float): Input parameter.

        Returns:
            float: Standard deviation.
        """
        return np.log(1.0 + np.exp(rho))

    def sigma_idt(self, rho):
        """Identity reparameterization for standard deviation.

            Args:
                rho (float): Input parameter.

            Returns:
                float: Standard deviation.
        """
        return rho + 0.0

    def get_posteriors(self, var_params):
        """Get the posterior distributions based on variational parameters.

        Args:
            var_params (np.ndarray): Variational parameters array of shape (2d,).

        Returns:
            list[pytuq.rv.mrv.MRV]: List of posterior distributions for each parameter.
        """
        assert(len(var_params) == 2 * self.pdim)
        means = var_params[:self.pdim]
        sigmas = [self.sigma(s) for s in var_params[self.pdim:]]

        # Set up the posteriors
        var_posteriors = []

        for ipar in range(self.pdim):
            var_posterior = Normal_1d(means[ipar], sigmas[ipar])
            var_posteriors.append(var_posterior)

        return var_posteriors


    def eval_loss_gmarg(self, var_params, ps=None):
        """Evaluate the negative log posterior with gauss-marginal approximation.

        Args:
            var_params (np.ndarray): Variational parameters array of shape (2d,).
            ps (np.ndarray, optional): Previous parameter samples of shape (nmc, pdim). Defaults to None.

        Returns:
            float: Negative log posterior value.
        """
        #print("Evaluating Loss ========================")
        # Set up the posteriors
        var_posteriors = self.get_posteriors(var_params)
        #print("Means ", [vp.mu for vp in var_posteriors])
        # Sample model parameters
        if ps is not None:
            #print("Grabbing previous parsample")
            self.par_samples = ps
        else:
            #print("Creating new parsample")
            self.par_samples = np.empty((self.nmc, self.pdim))
            for ipar in range(self.pdim):
                self.par_samples[:, ipar] = var_posteriors[ipar].sample(self.nmc)
            # need to trace back to xi_samples for grad computation

        # Evaluate model
        y_samples = self.model(self.par_samples)  # nmc, ndata

        ymean = np.mean(y_samples, axis=0)
        yvar = np.var(y_samples, axis=0)

        residual = ymean - self.y_data
        neg_log_likelihood = 0.5 * np.sum(np.log(2.0 * np.pi * (yvar + self.datasigma**2)))
        neg_log_likelihood += 0.5 * np.sum(residual**2 / (yvar + self.datasigma**2))

        # variational_logpost_samples = np.empty((self.nmc,))
        # log_prior_samples = np.empty((self.nmc,))
        # for imc in range(self.nmc):
        #     variational_logpost_samples[imc] = 0.0
        #     log_prior_samples[imc] = 0.0
        #     for ipar in range(self.pdim):
        #         variational_logpost_samples[imc] += (var_posteriors[ipar].logpdf(
        #             self.par_samples[imc, ipar]))  # - np.log(sigmas[ipar]) # this was needed if we started with N(0,1) posterior.
        #         log_prior_samples[imc] += self.priors[ipar].logpdf(
        #             self.par_samples[imc, ipar])

        # variational_logpost = variational_logpost_samples.mean()
        # log_prior = log_prior_samples.mean()

        # neg_log_likelihood = np.log(self.datasigma) + 0.5 * np.log(2.0 * np.pi) + \
        #     0.5 * ((y_samples - self.y_data)**2).mean() / self.datasigma**2
        # neg_log_likelihood *= self.y_data.shape[0]

        # var_loss = variational_logpost - log_prior + neg_log_likelihood

        log_prior = 0.0
        for ipar in range(self.pdim):
            log_prior += self.priors[ipar].logpdf(var_params[ipar])


        #print("Neg Log Likelihood: ", neg_log_likelihood, " Neg Log Prior: ", - log_prior)
        return -log_prior + neg_log_likelihood

    def eval_loss_abc(self, var_params, ps=None, eps=1.e-2, alpha=1.0):
        """Evaluate the negative log posterior with ABC approximation.

        Args:
            var_params (np.ndarray): Variational parameters array of shape (2d,).
            ps (np.ndarray, optional): Previous parameter samples of shape (nmc, pdim). Defaults to None.
            eps (float, optional): ABC tolerance. Defaults to 1.e-2.
            alpha (float, optional): Weighting factor for the ABC term. Defaults to 1.0.

        Returns:
            float: Negative log posterior value.
        """
        #print("Evaluating Loss ========================")
        # Set up the posteriors
        var_posteriors = self.get_posteriors(var_params)
        #print("Means ", [vp.mu for vp in var_posteriors])
        # Sample model parameters
        if ps is not None:
            #print("Grabbing previous parsample")
            self.par_samples = ps
        else:
            #print("Creating new parsample")
            self.par_samples = np.empty((self.nmc, self.pdim))
            for ipar in range(self.pdim):
                self.par_samples[:, ipar] = var_posteriors[ipar].sample(self.nmc)
            # need to trace back to xi_samples for grad computation

        # Evaluate model
        y_samples = self.model(self.par_samples)  # nmc, ndata

        ymean = np.mean(y_samples, axis=0)
        yvar = np.var(y_samples, axis=0)

        residual = ymean - self.y_data

        neg_log_likelihood = 0.0
        for idata in range(len(self.y_data)):
            neg_log_likelihood += 0.5 * (residual[idata]**2) / eps**2
            neg_log_likelihood += 0.5 * (alpha*np.abs(residual[idata])-np.sqrt(yvar[idata]+self.datasigma[idata]**2))**2 / eps**2
        neg_log_likelihood += 0.5 * np.log(2.*np.pi)
        neg_log_likelihood += np.log(eps)

        log_prior = 0.0
        for ipar in range(self.pdim):
            log_prior += self.priors[ipar].logpdf(var_params[ipar])


        #print("Neg Log Likelihood: ", neg_log_likelihood, " Neg Log Prior: ", - log_prior)
        return -log_prior + neg_log_likelihood

    def eval_loss_elbo(self, var_params, ps=None):
        """Evaluate the ELBO.

        Args:
            var_params (np.ndarray): Variational parameters array of shape (2d,).
            ps (np.ndarray, optional): Previous parameter samples of shape (nmc, pdim). Defaults to None.

        Returns:
            float: Negative log posterior value.
        """
        #print("Evaluating Loss ========================")
        # Set up the posteriors
        var_posteriors = self.get_posteriors(var_params)
        #print("Means ", [vp.mu for vp in var_posteriors])
        # Sample model parameters
        if ps is not None:
            #print("Grabbing previous parsample")
            self.par_samples = ps
        else:
            #print("Creating new parsample")
            self.par_samples = np.empty((self.nmc, self.pdim))
            for ipar in range(self.pdim):
                self.par_samples[:, ipar] = var_posteriors[ipar].sample(self.nmc)
            # need to trace back to xi_samples for grad computation

        # Evaluate model
        y_samples = self.model(self.par_samples)  # nmc, ndata

        variational_logpost_samples = np.empty((self.nmc,))
        log_prior_samples = np.empty((self.nmc,))
        for imc in range(self.nmc):
            variational_logpost_samples[imc] = 0.0
            log_prior_samples[imc] = 0.0
            for ipar in range(self.pdim):
                variational_logpost_samples[imc] += (var_posteriors[ipar].logpdf(
                    self.par_samples[imc, ipar]))  # - np.log(sigmas[ipar]) # this was needed if we started with N(0,1) posterior.
                log_prior_samples[imc] += self.priors[ipar].logpdf(
                    self.par_samples[imc, ipar])

        variational_logpost = variational_logpost_samples.mean()
        log_prior = log_prior_samples.mean()

        neg_log_likelihood = np.mean(np.log(self.datasigma)) + 0.5 * np.log(2.0 * np.pi) + \
            0.5 * (((y_samples - self.y_data)/self.datasigma)**2).mean()
        neg_log_likelihood *= self.y_data.shape[0]

        var_loss = variational_logpost - log_prior + neg_log_likelihood

        return var_loss

    def eval_loss_elbo_grad_(self, x, eps=1.e-3):
        """Evaluate the gradient of the ELBO using finite differences.

        Args:
            x (np.ndarray): Variational parameters array of shape (2d,).
            eps (float, optional): Finite difference step size. Defaults to 1.e-3.

        Returns:
            np.ndarray: Gradient of the ELBO.
        """
        #print("Evaluating Gradient ========================")
        ndim = len(x)
        grad = np.zeros((ndim,))
        for idim in range(ndim):
            #print("Dim ", idim)
            xx2 = x.copy()
            xx2[idim] += eps
            xx1 = x.copy()
            xx1[idim] -= eps
            grad[idim] = (self.eval_loss(xx2, ps=self.par_samples) -
                          self.eval_loss(xx1, ps=self.par_samples)) / (2. * eps)
        return grad

    def plot_parpdf(self, var_params, ax=None):
        """Plot the variational posterior PDFs.

        Args:
            var_params (np.ndarray): Variational parameters array of shape (2d,).
            ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None which plots on the current axis.
        """
        if ax is None:
            ax = plt.gca()
        var_posteriors = self.get_posteriors(var_params)

        for ipar, posterior in enumerate(var_posteriors):
            xgrid = np.linspace(posterior.mu - 5 * posterior.sigma,
                                posterior.mu + 5 * posterior.sigma, 101)
            ygrid = posterior.pdf(xgrid)

            _ = plt.figure(figsize=(12, 8))
            plt.plot(xgrid, ygrid, '-', label='VI Posterior PDF')
            plt.xlabel(f'Param #{ipar}')
            plt.savefig(f'post_vi_{ipar}.png')
            plt.clf()

    def compute_pred(self, var_params, nsam, model=None):
        """Compute model predictions based on variational parameters.

        Args:
            var_params (np.ndarray): Variational parameters array of shape (2d,).
            nsam (int): Number of samples to draw.
            model (callable, optional): Forward model. Defaults to None, which uses self.model.

        Returns:
            np.ndarray: Model prediction samples of shape (nsam, ndata).
        """
        par_samples = self.sample_posterior(var_params, nsam)

        if model is None:
            model = self.model

        y_samples = model(par_samples)

        return y_samples


    def sample_posterior(self, var_params, nsam):
        """Sample from the variational posterior.

        Args:
            var_params (np.ndarray): Variational parameters array of shape (2d,).
            nsam (int): Number of samples to draw.

        Returns:
            np.ndarray: Samples from the variational posterior of shape (nsam, pdim).
        """
        var_posteriors = self.get_posteriors(var_params)

        par_samples = np.empty((nsam, self.pdim))
        for ipar in range(self.pdim):
            par_samples[:, ipar] = var_posteriors[ipar].sample(nsam)

        return par_samples
