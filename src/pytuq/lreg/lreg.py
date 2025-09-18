#!/usr/bin/env python
"""Module for the base linear regression class and a bare minimum least-squares implementation."""

import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

from ..fit.fit import fitbase
from ..utils.plotting import plot_dm


class lreg(fitbase):
    """Base class for linear regression.

    Attributes:
        cf (np.ndarray): An 1d array of coefficients, of size :math:`K`.
        cf_cov (np.ndarray): A 2d array of coefficient covariance of size :math:`(K,K)`.
        datavar (float): A single value for homogenous data variance.
        basisEval (callable): Basis evaluator function.
        basisEvalPars (tuple): Parameters of the basis evaluator function.
        basisEvaluatorSet (bool): Indicates whether basis evaluator function is set.
        used (np.ndarray): Indicates the indices of the used bases (for sparse learning, such as BCS).
    """

    def __init__(self):
        """Initialization."""
        super().__init__()

        self.cf = None
        self.cf_cov = None
        self.datavar=0.0
        self.basisEvaluatorSet = False
        self.used = None



    def setBasisEvaluator(self, basiseval, basisevalpars):
        """Setting basis evaluator function.

        Args:
        basiseval (callable): Basis evaluator function.
        basisevalpars (tuple): Parameters of the basis evaluator function.

        Returns:
            TYPE: Description
        """
        self.basisEval = basiseval
        self.basisEvalPars = basisevalpars
        self.basisEvaluatorSet = True

        return

    def fita(self, Amat, y):
        r"""Fitting function. Not implemented in the base class.

        Args:
            Amat (np.ndarray): A 2d array of size :math:`(N, K)` each row holding basis evaluations at a training point.
            y (np.ndarray): An 1d array of size :math:`N` holding the data.

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError("Fitting not implemented in the base class.")


    def fit(self, x, y):
        r"""Fitting with :math:`(x,y)` pairs, assuming basis evaluator is set.

        Args:
            x (np.ndarray): A 2d array of inputs at which bases are evaluated.
            y (np.ndarray): An 1d array of data.
        """
        assert(self.basisEvaluatorSet)
        Amat = self.basisEval(x, self.basisEvalPars)
        self.fita(Amat, y)


    def print_coefs(self):
        """Prints coefficients, assuming fit is already performed."""
        assert(self.fitted)
        print(self.cf)


    def predict(self, x, msc=0, pp=False):
        r"""Predict function, given input :math:`x`, assuming the basis evaluator is set.

        Args:
            x (np.ndarray): A 2d array of inputs of size :math:`(N,d)` at which bases are evaluated.
            msc (int, optional): Prediction mode: 0 (mean-only), 1 (mean and variance), or 2 (mean, variance and covariance). Defaults to 0.
            pp (bool, optional): Whether to compute posterior-predictive (i.e. add data variance) or not.

        Returns:
            tuple(np.ndarray, np.ndarray, np.ndarray): triple of Mean (array of size `N`), Variance (array of size `N` or None), Covariance (array of size `(N, N)` or None).
        """
        assert(self.basisEvaluatorSet)
        Amat = self.basisEval(x, self.basisEvalPars)

        return self.predicta(Amat, msc=msc, pp=pp)


    def predicta(self, Amat, msc=0, pp=False):
        r"""Predict given the A-matrix of basis evaluations.

        Args:
            Amat (np.ndarray): A 2d array of size :math:`(N, K)` each row holding basis evaluations at a training point.
            msc (int, optional): Prediction mode: 0 (mean-only), 1 (mean and variance), or 2 (mean, variance and covariance). Defaults to 0.
            pp (bool, optional): Whether to compute posterior-predictive (i.e. add data variance) or not.

        Returns:
            tuple(np.ndarray, np.ndarray, np.ndarray): triple of Mean (array of size `N`), Variance (array of size `N` or None), Covariance (array of size `(N, N)` or None).
        """
        assert(self.fitted)

        ypred = Amat @ self.cf
        if msc==2:
            ypred_cov = (Amat @ self.cf_cov) @ Amat.T + int(pp)*self.datavar*np.eye(Amat.shape[0])
            ypred_var = np.diag(ypred_cov)
        elif msc==1:
            ypred_cov = None
            try:
                ypred_var = self.compute_stdev(Amat, method='chol')**2
            except np.linalg.LinAlgError:
                ypred_var = self.compute_stdev(Amat, method='svd')**2

            ypred_var += int(pp)*self.datavar
        elif msc==0:
            ypred_cov = None #np.zeros((Amat.shape[1], Amat.shape[1]))
            ypred_var = None #np.zeros((Amat.shape[1],))
        else:
            print(f"msc={msc}, but needs to be 0,1, or 2. Exiting.")
            sys.exit()

        return ypred, ypred_var, ypred_cov

    def compute_stdev(self, Amat, method="chol"):
        r"""Computation of pushed-forward standard deviation using a few different methods.

        Args:
            Amat (np.ndarray): A 2d array of size :math:`(N, K)` each row holding basis evaluations at a training point.
            method (str, optional): Method of computation. Options are 'chol' (cholesky), 'choleye' (cholesky with regularization), 'svd' (SVD), 'loop' (slow and painful), 'fullcov' (slow-ish and memory intensive).

        Returns:
            np.ndarray: An 1d array of pushed-forward standard deviation, of size :math:`N`.
        """
        assert(self.cf_cov is not None)
        if method == "chol":
            chol = np.linalg.cholesky(self.cf_cov)
            mat = Amat @ chol
            pf_stdev = np.linalg.norm(mat, axis=1)
        elif method == "choleye":
            eigvals = np.linalg.eigvalsh(self.cf_cov)
            chol = np.linalg.cholesky(self.cf_cov+(abs(eigvals[0]) + 1e-14) * np.eye(self.cf_cov.shape[0]))
            mat = Amat @ chol
            pf_stdev = np.linalg.norm(mat, axis=1)
        elif method == "svd":
            u, s, vh = np.linalg.svd(self.cf_cov, hermitian=True)
            mat = (Amat @ u) @ np.sqrt(np.diag(s))
            pf_stdev = np.linalg.norm(mat, axis=1)
        elif method == "loop":
            tmp = np.dot(Amat, self.cf_cov)
            pf_stdev = np.empty(Amat.shape[0])
            for ipt in range(Amat.shape[0]):
                pf_stdev[ipt] = np.sqrt(np.dot(tmp[ipt, :], Amat[ipt, :]))
        elif method == "fullcov":
            pf_stdev = np.sqrt(np.diag((Amat @ self.cf_cov) @ Amat.T))
        else:
            pf_stdev = np.zeros(Amat.shape[0])

        return pf_stdev


    def predict_plot(self, aa_list, yy_list, labels=None, colors=None):
        r"""Ploting utility given a list of A-matrices and a list of data (typically, training/validation/testing).

        Args:
            aa_list (list[np.ndarray]): list of A-matrices of size :math:`(\cdot, d)`.
            yy_list (list[np.ndarray]): list of 1d data arrays
            labels (None, optional): list of labels, of the same size as the lists above. Defaults to None, i.e. no labels.
            colors (None, optional): list of colors, of the same size as the lists above. Defaults to None, i.e. select standard colors.
        """
        nlist = len(aa_list)
        assert(nlist==len(yy_list))

        yy_pred_list = []
        yy_pred_std_list = []
        for aa in aa_list:
            yy_pred, yy_pred_var, _ = self.predicta(aa, msc=1)
            yy_pred_list.append(yy_pred)
            yy_pred_std_list.append(np.sqrt(yy_pred_var))

        if labels is None:
            labels = [f'Set {i+1}' for i in range(nlist)]
        assert(len(labels)==nlist)

        if colors is None:
            colors = ['b', 'g', 'r', 'c', 'm', 'y']*nlist
            colors = colors[:nlist]
        assert(len(colors)==nlist)

        ee = list(zip(yy_pred_std_list, yy_pred_std_list))

        plot_dm(yy_list, yy_pred_list, errorbars=ee, labels=labels, colors=colors,
                axes_labels=[f'Data', f'Fit'],
                figname='fitdiag.png',
                legendpos='in', msize=13)

        res_list = [np.abs(y_pred-y) for y, y_pred in zip(yy_list, yy_pred_list)]
        plot_dm(res_list, yy_pred_std_list, errorbars=None, labels=labels, colors=colors,
                axes_labels=[f'|Residual|', f'Fit StDev'],
                figname='resid_vs_std.png',
                legendpos='in', msize=13)

        _ = plt.figure(figsize=(12,8))
        for y, y_pred_std, label, color in zip(yy_list, yy_pred_std_list, labels, colors):
            ind = np.argsort(y)
            plt.plot(y[ind], y_pred_std[ind], 'o', color=color, label=label)
        plt.legend()
        plt.xlabel('Data')
        plt.ylabel('Fit StDev')
        plt.savefig('y_ystd.png')
        plt.yscale('log')
        plt.savefig('y_ystd_log.png')

        _ = plt.figure(figsize=(12,8))
        for y_pred, y_pred_std, label, color in zip(yy_pred_list, yy_pred_std_list, labels, colors):
            ind = np.argsort(y_pred)
            plt.plot(y_pred[ind], y_pred_std[ind], 'o', color=color, label=label)
        plt.legend()
        plt.xlabel('Fit Mean')
        plt.ylabel('Fit StDev')
        plt.savefig('ypred_ystd.png')
        plt.yscale('log')
        plt.savefig('ypred_ystd_log.png')

        for y, y_pred, y_pred_std, label, color in zip(yy_list, yy_pred_list, yy_pred_std_list, labels, colors):
            _ = plt.figure(figsize=(12,8))
            plt.plot([0, len(y)], [1.0, 1.0], 'r--')
            plt.plot(y_pred_std/np.abs(y-y_pred), 'o', color=color)
            plt.xlabel(label + ' Sample')
            plt.ylabel('Fit StDev / |Residual|')
            plt.savefig('ystd_resid_'+label+'.png')
            plt.yscale('log')
            plt.savefig('ystd_resid_'+label+'_log.png')

        for y, y_pred, y_pred_std, label, color in zip(yy_list, yy_pred_list, yy_pred_std_list, labels, colors):
            _ = plt.figure(figsize=(12,8))
            plt.plot([0, len(y)], [1.0, 1.0], 'r--')
            plt.plot(np.abs(y-y_pred)/y_pred_std, 'o', color=color)
            plt.xlabel(label + ' Sample')
            plt.ylabel('|Residual| / Fit StDev')
            plt.savefig('resid_ystd_'+label+'.png')
            plt.yscale('log')
            plt.savefig('resid_ystd_'+label+'_log.png')

        # for y, y_pred, y_pred_std, label, color in zip(yy_list, yy_pred_list, yy_pred_std_list, labels, colors):
        #     _ = plt.figure(figsize=(10,10))
        #     plt.plot(np.abs(y-y_pred), y_pred_std, 'o', color=color)
        #     plt.xlabel('|Residual|')
        #     plt.ylabel('Fit StDev')
        #     plt.savefig('resid_vs_ystd_'+label+'.png')
        #     plt.xscale('log')
        #     plt.yscale('log')
        #     plt.savefig('resid_vs_ystd_'+label+'_log.png')

        for y, y_pred, y_pred_std, label, color in zip(yy_list, yy_pred_list, yy_pred_std_list, labels, colors):
            _ = plt.figure(figsize=(12,8))
            plt.plot((y_pred-y)/y_pred_std, 'o', color=color, markeredgecolor='w')
            plt.fill_between([0, len(y)], [-1.0, -1.0], [1.0, 1.0], color='grey', alpha=1.0, label=r'$1\sigma$ region')
            plt.xlabel(label + ' Sample')
            plt.ylabel('Residual / Fit StDev')
            if label == 'Training':
                plt.ylim([-7., 7.])
            elif label == 'Testing':
                plt.ylim([-10., 10.])
            plt.legend()
            plt.savefig('resid_scaled_'+label+'.png')




class lsq(lreg):
    """Bare minimum least squares solution.

    Attributes:
        cf (np.ndarray): An 1d array of coefficients, of size :math:`K`.
        cf_cov (np.ndarray): A 2d array of coefficient covariance of size :math:`(K,K)`.
        fitted (bool): Flag to indicate whether fit is performed or not.
        used (np.ndarray): An array of integers indicating the bases used (i.e. all basis in this case).

    Note:
        scipy's lstsq uses SVD under the hood.
    """

    def __init__(self):
        """Initialization."""
        super().__init__()


    def fita(self, Amat, y):
        r"""Fit given A-matrix of basis evaluations and data array.

        Args:
            Amat (np.ndarray): A 2d array of size :math:`(N, K)` each row holding basis evaluations at a training point.
            y (np.ndarray): An 1d array of size :math:`N` holding the data.
        """
        self.cf, residues, rank, s = lstsq(Amat, y, 1.0e-13)
        self.cf_cov = np.zeros((Amat.shape[1], Amat.shape[1]))
        self.fitted = True

        self.used = np.arange(Amat.shape[1])
