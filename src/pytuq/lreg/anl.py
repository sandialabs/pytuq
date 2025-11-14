#!/usr/bin/env python
"""Analytically available Bayesian linear regression."""

import sys
import numpy as np

from .lreg import lreg



class anl(lreg):
    """Bayesian linear regression class.

    Attributes:
        cf (np.ndarray): An 1d array of coefficients, of size :math:`K`.
        cf_cov (np.ndarray): A 2d array of coefficient covariance of size :math:`(K,K)`.
        cov_nugget (float): A diagonal covariance nugget to regularize the matrix inversion.
        datavar (float): A single value for homogenous data variance.
        fitted (bool): Flag to indicate whether fit is performed or not.
        method (string): Method of fitting. Can be 'full' (the conventional Bayesian linear regression) or 'vi' (the variational approximation, still available analytically).
        prior_var (float): A single value for homogenous prior variance.
        used (np.ndarray): An array of integers indicating the bases used (i.e. all basis in this case).
    """

    def __init__(self, datavar=None, prior_var=None, cov_nugget=0.0, method='full'):
        """Initialization.

        Args:
            method (string, optional): Method of fitting. Can be 'full' (the conventional Bayesian linear regression) or 'vi' (the variational approximation, still available analytically). Defaults to 'full'.
            datavar (float, optional): A single value for homogenous data variance. Defaults to None, which means compute the optimal value.
            prior_var (float, optional): A single value for homogenous prior variance. Defaults to None, which means ignore the prior variance (infinite variance).
            cov_nugget (float, optional): A diagonal covariance nugget to regularize the matrix inversion. Defaults to 0.0.
        """
        super().__init__()
        self.prior_var = prior_var
        self.datavar = datavar
        self.cov_nugget = cov_nugget
        self.method = method

        if self.prior_var is not None:
            assert(self.datavar is not None)
            assert(self.method == 'full')
            # TODO: need to do the math and implement


    def _compute_sigmahatsq(self, Amat, y):
        r"""Computes best variance estimate.

        Args:
            Amat (np.ndarray): A 2d array of size :math:`(N, K)` each row holding basis evaluations at a training point.
            y (np.ndarray): An 1d array of size :math:`N` holding the data.

        Returns:
            float: Best estimate of variance, :math:`\hat{\sigma}^2`.
        """
        npt, nbas = Amat.shape
        bp = np.dot(y, y - np.dot(Amat, self.cf))/2.

        ap = (npt - nbas)/2.
        sigmahatsq = bp/(ap-1.)

        # TODO: isn't it better to have a warning or error here (or elsewhere?)
        if sigmahatsq < 0.0:
            sigmahatsq = 0.0

        return sigmahatsq

    def fita(self, Amat, y):
        r"""Fit given A-matrix of basis evaluations and data array.

        Args:
            Amat (np.ndarray): A 2d array of size :math:`(N, K)` each row holding basis evaluations at a training point.
            y (np.ndarray): An 1d array of size :math:`N` holding the data.
        """
        npt, nbas = Amat.shape

        ptp = np.dot(Amat.T, Amat)
        if self.prior_var is not None:
            ptp += (self.datavar/self.prior_var) * np.eye(nbas)
        ptp += self.cov_nugget*np.eye(nbas)
        invptp = np.linalg.inv(ptp)
        self.cf = np.dot(invptp, np.dot(Amat.T, y))
        if self.datavar is None:
            sigmahatsq = self._compute_sigmahatsq(Amat, y)
        else:
            sigmahatsq = self.datavar

        self.datavar = sigmahatsq+0.0


        #print("Datanoise variance : ", sigmahatsq)

        # True posterior covariance
        if self.method == 'full':
            self.cf_cov = sigmahatsq*invptp
            np.savetxt('covar.txt', self.cf_cov)
        # Variational covariance
        elif self.method == 'vi':
            self.cf_cov = np.diag(sigmahatsq/np.diag(np.dot(Amat.T, Amat)+self.cov_nugget*np.diag(np.ones((nbas,)))))
        else:
            print(f"Method {self.method} unknown. Exiting.")
            sys.exit()

        self.fitted = True
        self.used = np.arange(Amat.shape[1])



    def compute_evidence(self, Amat, ydata):
        """Compute the evidence.

        Args:
            Amat (np.ndarray): A 2d array of size :math:`(N, K)` each row holding basis evaluations at a training point.
            ydata (np.ndarray): An 1d array of size :math:`N` holding the data.

        Returns:
            float: log-evidence value

        Note: see https://krasserm.github.io/2019/02/23/bayesian-linear-regression/ Eq (18) for the closest (and equivalent) formula to the one implemented.
        """
        assert(self.prior_var is not None)
        assert(self.fitted)

        npt, nbas = Amat.shape
        ptp = np.dot(Amat.T, Amat)

        det_cf_cov = np.linalg.det(self.cf_cov)

        if det_cf_cov <= 0.0 or self.datavar <= 0.0:
            return -sys.float_info.max

        evid = 0.5*np.log(det_cf_cov)

        evid += -0.5*nbas*np.log(self.prior_var)
        evid += -0.5*npt*np.log(2.*np.pi*self.datavar)
        evid += -(np.dot(ydata,ydata)-self.datavar*np.dot(self.cf,np.dot((1./self.datavar)*ptp+(1./self.prior_var)*np.eye(nbas),self.cf)))/(2.*self.datavar)

        return evid
