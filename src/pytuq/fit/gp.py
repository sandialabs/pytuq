#!/usr/bin/env python
"""Module for Gaussian Process regression."""

import sys
import numpy as np
from scipy.linalg import cho_solve, solve
from scipy.optimize import minimize

from .fit import fitbase

def kernel_rbf(x, y, corlength):
    r"""Radial basis function kernel :math:`K(x,y)=e^{-\frac{||x-y||^2}{2 l^2}}`.

    Args:
        x (np.ndarray): 1d array of size `d`.
        y (np.ndarray): 1d array of size `d`.
        corlength (float): Correlation length.

    Returns:
        float: Kernel value.
    """
    return np.exp(-0.5*(np.linalg.norm(x-y)/corlength)**2)

def kernel_sin(x, y, corlength, period):
    r"""Sinusoidal kernel :math:`K(x,y)=e^{-\frac{2}{l^2} \sin^2\left({\frac{\pi ||x-y||}{T}}\right)}`.

    Args:
        x (np.ndarray): 1d array of size `d`.
        y (np.ndarray): 1d array of size `d`.
        corlength (float): Correlation length.

    Returns:
        float: Kernel value.
    """
    return np.exp((-2./corlength**2)*np.sin(np.pi*np.linalg.norm(x-y)/period)**2)

class gp(fitbase):
    r"""Gaussian process class.

    Attributes:
        nbas (int): Number of regression bases, :math:`K`.
        basisEval (callable): Basis evaluator function of signature :math:`f(x,p_{bas})`, where :math:`x` is a 2d array of size `(N,d)` and output is a 2d array of size :math:`(N, K)`.
        basisEvalPars (list): Parameters :math:`p_{bas}` of the basis evaluator.
        basisEvaluatorSet (bool): Indicates whether the basis evaluator is set or not.
        AinvH (np.ndarray): A 2d matrix :math:`A^{-1}H` of size :math:`(N,K)`.
        c_hat (np.ndarray): The best regression coefficient, an 1d array :math:`\hat{c}` of size :math:`K`.
        fitted (bool): Indicates whether the GP is already built.
        kernel (callable): Kernel evaluator function of signature :math:`K(x_1, x_2, p_{ker})`, where :math:`x_1` and :math:`x_2` are 1d arrays of size :math:`d`, and the output is a scalar.
        kernel_params (list): Parameters :math:`p_{ker}` of the kernel.
        kernel_params_range (list[tuple]): List of (min, max) tuples of size of kernel parameter list.
        LVst (np.ndarray): Cholesky factor of :math:`V^*`, a 2d array of size :math:`(K,K)`.
        Vstinv (np.ndarray): Inverse of :math:`V^*`, a 2d array of size :math:`(K,K)`.
        sigma2 (float): Data variance, given or inferred.
        nugget (float): Covariance nugget :math:`\epsilon` to improve conditioning.
        prior_invcov (np.ndarray): Inverse-covariance of the prior, a 2d array of size :math:`(K,K)`.
        prior_mean (np.ndarray): Mean of the prior, a 1d array of size :math:`(K,K)`.
        kf_ (np.ndarray): An 1d array :math:`k_f` of size :math:`N`.
        L_ (np.ndarray): Cholesky factor, a 2d array of size :math:`(N,N)`.
        x_ (np.ndarray): An 2d array of size :math:`(N,d)`, the training x-data.
        y_ (np.ndarray): An 1d array of size :math:`N`, the training y-data.
    """

    def __init__(self, kernel, kernel_params, kernel_params_range=None, sigma2=None, nugget=0.0, sigma2prior=None, prior_mean=None, prior_invcov=None, basis=None):
        r"""Initialization.

        Args:
            kernel (callable): Kernel evaluator function of signature :math:`K(x_1, x_2, p_{ker})`, where :math:`x_1` and :math:`x_2` are 1d arrays of size :math:`d`, and the output is a scalar.
            kernel_params (list): Parameters :math:`p_{ker}` of the kernel.
            kernel_params_range (list[tuple], optional): List of (min, max) tuples of size of kernel parameter list. Defaults to None, i.e. no bounds in optimizing kernel parameters.
            sigma2 (float): Data variance, given or inferred.
            nugget (float): Covariance nugget :math:`\epsilon` to improve conditioning.
            prior_invcov (np.ndarray): Inverse-covariance of the prior, a 2d array of size :math:`(K,K)`.
            prior_mean (np.ndarray): Mean of the prior, a 1d array of size :math:`(K,K)`.
            basis (tuple, optional): A tuple of (Basis evaluator, Basis evaluator parameters). Defaults to None, in which case no basis (i.e. no regression) is used.
            sigma2prior (tuple, optional): A tuple of :math:`(\alpha, \beta)` parameters of data variance prior. Defaults to None, in which case both parameters are set to zero, corresponding to prior :math:`p(\sigma^2)=1/\sigma^2`.
        """
        super().__init__()

        self.kernel = kernel
        self.kernel_params = kernel_params
        self.kernel_params_range = kernel_params_range

        self.sigma2 = sigma2
        self.nugget = nugget

        if sigma2prior is not None:
            assert(sigma2 is None)
            self.alpha, self.beta = sigma2prior
        else:
            self.alpha, self.beta = 0.0, 0.0

        self.prior_mean = prior_mean
        self.prior_invcov = prior_invcov

        self.nbas = 0
        self.basisEvaluatorSet = False

        if basis is not None:
            basiseval, basisevalpars = basis
            self.setBasisEvaluator(basiseval, basisevalpars)


    def setBasisEvaluator(self, basiseval, basisevalpars):
        """Setting basis evaluator.

        Args:
            basiseval (callable): Basis evaluator function of signature :math:`f(x,p_{bas})`, where :math:`x` is a 2d array of size `(N,d)` and output is a 2d array of size :math:`(N, K)`.
            basisevalpars (list): Parameters :math:`p_{bas}` of the basis evaluator.
        """
        self.basisEval = basiseval
        self.basisEvalPars = basisevalpars
        self.basisEvaluatorSet = True

    def set_prior(self, nbas):
        """Setting the prior on basis coefficients.

        Args:
            nbas (int): Number of bases
        """
        if self.prior_mean is None:
            self.prior_mean=np.zeros(nbas,)
        if self.prior_invcov is None:
            self.prior_invcov=np.zeros((nbas,nbas))

    def get_sigma2hat(self, y):
        r"""Get best variance.

        Args:
            y (np.ndarray): An 1d array of training y-data of size :math:`N`.

        Returns:
            float: Best data variance, :math:`\hat{\sigma}^2`.
        """
        npt = y.shape[0]
        assert(self.fitted)
        assert(self.sigma2 is None)
        sigma2 = 2.*self.beta+np.dot(y, self.kf_)

        if self.basisEvaluatorSet:
            sigma2 += np.dot(self.prior_mean, self.prior_invcov @ self.prior_mean)
            sigma2 -= np.dot(self.c_hat, self.Vstinv @ self.c_hat)


        sigma2 /= (npt+2.*self.alpha-self.nbas-2.)

        print("Best data variance : ", sigma2)
        self.sigma2 = sigma2.copy()

        return sigma2

    def fit(self, x, y):
        """Fitting the GP.

        Args:
            x (np.ndarray): An 2d array of size :math:`(N,d)`, the training x-data.
            y (np.ndarray): An 1d array of size :math:`N`, the training y-data.

        """
        self.x_ = x
        self.y_ = y

        self.set_kernel(kernel=self.kernel, kernel_params=self.kernel_params, kernel_params_range=self.kernel_params_range)

        Kmat = self.get_kmat_self(x)

        self.L_ = np.linalg.cholesky(Kmat+self.nugget*np.eye(x.shape[0]))
        self.kf_ = cho_solve((self.L_, True), y)

        if self.basisEvaluatorSet:

            Hmat = self.basisEval(x, self.basisEvalPars)
            self.nbas = Hmat.shape[1]

            self.set_prior(self.nbas)

            self.AinvH = cho_solve((self.L_, True), Hmat)
            tmp = Hmat.T @ self.AinvH
            self.Vstinv = self.prior_invcov+tmp
            Vst = np.linalg.pinv(self.Vstinv)
            self.c_hat = Vst @ (np.dot(self.prior_invcov, self.prior_mean) + Hmat.T @ self.kf_)
            self.LVst = np.linalg.cholesky(Vst)

            AinvH_LVst = self.AinvH @ self.LVst
            # Cb = AinvH_LVst @ AinvH_LVst.T

        self.fitted = True
        _ = self.get_sigma2hat(y)

    def predict(self, xc, msc=0, pp=False):
        r"""Predict function, given input :math:`x`, assuming the GP is built.

        Args:
            xc (np.ndarray): A 2d array of inputs of size :math:`(N,d)` at which bases are evaluated.
            msc (int, optional): Prediction mode: 0 (mean-only), 1 (mean and variance), or 2 (mean, variance and covariance). Defaults to 0.
            pp (bool, optional): Whether to compute posterior-predictive (i.e. add data variance) or not.

        Returns:
            tuple(np.ndarray, np.ndarray, np.ndarray): triple of Mean (array of size :math:`N`), Variance (array of size :math:`N` or None), Covariance (array of size :math:`(N, N)` or None).
        """
        assert(self.fitted)


        Kcross = self.get_kmat_cross(xc)
        Ktest = self.get_kmat_self(xc)

        yc = Kcross @ self.kf_
        if self.basisEvaluatorSet:

            Hmat = self.basisEval(xc, self.basisEvalPars)
            yc += (Hmat - Kcross @ self.AinvH ) @ self.c_hat

        if msc>0: # TODO: need to do var in a fast way and differ msc=1 and msc=2
            v = cho_solve((self.L_, True), Kcross.T)
            yc_cov = Ktest - Kcross @ v
            if self.basisEvaluatorSet:
                tmp = (Hmat - Kcross @ self.AinvH) @ self.LVst
                yc_cov += tmp @ tmp.T

            yc_cov *= self.sigma2
            yc_cov += int(pp)*self.sigma2*np.eye(yc_cov.shape[0])

            yc_var = np.diag(yc_cov)

        else:
            yc_var = yc_cov = None

        return yc, yc_var, yc_cov

    def set_kernel(self, kernel, kernel_params, kernel_params_range=None):
        """Set the kernel function.

        Args:
            kernel (callable): Kernel evaluator function of signature :math:`K(x_1, x_2, p_{ker})`, where :math:`x_1` and :math:`x_2` are 1d arrays of size :math:`d`, and the output is a scalar.
            kernel_params (list): Parameters :math:`p_{ker}` of the kernel.
            kernel_params_range (list[tuple], optional): List of (min, max) tuples of size of kernel parameter list. Defaults to None, i.e. no bounds in optimizing kernel parameters.
        """
        if isinstance(kernel, str):
            if kernel == 'RBF':
                self.kernel = kernel_rbf
            elif kernel == 'sin':
                self.kernel = kernel_sin
            else:
                print(f"Kernel {kernel} is unknown. Exiting.")
                sys.exit()

        if kernel_params_range is not None:
            res = minimize(self.neglogmarglik, kernel_params, args=(self.x_,self.y_), method='L-BFGS-B', jac=None, tol=None, bounds=kernel_params_range, callback=None, options=None)
            self.kernel_params = res.x


    def get_kmat_self(self, x, kernel_params=None):
        r"""Get the self-matrix of kernel evaluations.

        Args:
            x (np.ndarray): A 2d array of inputs of size :math:`(N,d)` at which kernel is evaluated.
            kernel_params (list): Parameters :math:`p_{ker}` of the kernel.

        Returns:
            np.ndarray: Kernel matrix, a 2d array of size :math:`(N,N)`.
        """
        if kernel_params is None:
            kernel_params = self.kernel_params

        npt, ndim = x.shape

        Kmat = np.empty((npt,npt))
        for i in range(npt):
            for j in range(npt):
                Kmat[i,j] = self.kernel(x[i, :], x[j, :], *kernel_params)

        return Kmat

    def get_kmat_cross(self, xc, kernel_params=None):
        r"""Get the cross-matrix of kernel evaluations of given points crossed with the training points.

        Args:
            xc (np.ndarray): A 2d array of inputs of size :math:`(N,d)` at which kernel is evaluated.
            kernel_params (list): Parameters :math:`p_{ker}` of the kernel.

        Returns:
            np.ndarray: Kernel matrix, a 2d array of size :math:`(N,N_{tr})`.
        """
        if kernel_params is None:
            kernel_params = self.kernel_params


        nptx, ndim = self.x_.shape
        nptxc, ndim_ = xc.shape
        assert(ndim==ndim_)

        Kcross = np.empty((nptxc, nptx))
        for i in range(nptxc):
            for j in range(nptx):
                Kcross[i,j] = self.kernel(xc[i, :], self.x_[j, :], *kernel_params)

        return Kcross

    def neglogmarglik(self, params, x, y):
        """Evaluates negative marginal log-likelihood.

        Args:
            params (list): Parameters :math:`p_{ker}` of the kernel to be optimized.
            x (np.ndarray): An 2d array x-data of size :math:`(N,d)`.
            y (np.ndarray): An 1d array y-data of size :math:`N`.

        Returns:
            float: The scalar value of the negative log-likelihood.
        """
        N = x.shape[0]
        Kmat = self.get_kmat_self(x, kernel_params=params)
        L_ = np.linalg.cholesky(Kmat+self.nugget*np.eye(N))

        value = 0.0
        prior_resid = y.copy()
        if self.basisEvaluatorSet:
            Hmat = self.basisEval(x, self.basisEvalPars)
            nbas = Hmat.shape[1]
            self.set_prior(nbas)
            prior_resid -= Hmat @ self.prior_mean

            AinvH = cho_solve((L_, True), Hmat)
            Vst = np.linalg.pinv(self.prior_invcov+Hmat.T @ AinvH)
            LVst = np.linalg.cholesky(Vst)
            AinvH_LVst = AinvH @ LVst

            tmp = prior_resid @ AinvH_LVst
            value = 0.5*tmp @ tmp.T
            value += np.sum(np.log(np.diag(LVst)))

            if np.linalg.norm(self.prior_invcov)>1.e-12:
                value += 0.5*np.log(np.linalg.invcov)
            else:
                value += 0.5*nbas*np.log(2.*np.pi)

        alpha_ = cho_solve((L_, True), prior_resid).reshape(-1, 1)

        value -= 0.5*N*np.log(2.*np.pi)
        value -= 0.5*np.dot(prior_resid, alpha_)

        value -= np.sum(np.log(np.diag(L_)))

        return -value


