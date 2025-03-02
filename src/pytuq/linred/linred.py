#!/usr/bin/env python
"""Module for parent linear dimensionality reduction class."""


import numpy as np
import matplotlib.pyplot as plt

from ..utils.plotting import myrc

myrc()

class LinRed(object):
    r"""Class for linear dimensionality reduction.

    Attributes:
        eigval (np.ndarray): An array of size `K` storing eigenvalues.
        mean (np.ndarray): An array of size `N` storing the mean.
        modes (np.ndarray): A 2d array of size `(N,K)` consisting of all eigenvectors.
        xi (np.ndarray): An array of size `(M,K)` for latent features.
        built (bool): Flag to indicate if the dimensionality redution is built or not.

    Note:
        Unless truncation is done, `K=N`.
    """
    def __init__(self):
        """Initialization."""
        self.built = False
        self.mean = None
        self.eigval = None
        self.modes = None
        self.xi = None

    def build(self, data, plot=True):
        r"""Build method that should be implemented in children classes.

        Args:
            data (np.ndarray): Data array of size `(N,M)`.
            plot (bool, optional): Flag indicating if auxilliary figures need to be made.

        Raises:
            NotImplementedError: Not implemented in this parent class.
        """
        raise NotImplementedError

    def project(self, data, subtract_mean=True):
        r"""Projecting data to the built bases.

        Args:
            data (np.ndarray): Data array of size `(N,M)`.
            subtract_mean (bool, optional): Whether to subtract the mean before projection or not. Defaults to True. Useful without subtraction, e.g. if we are projecting perturbations or standard deviations.

        Returns:
            np.ndarray: Array of latent features of size `(N,K)`.
        """
        assert(self.built)
        xi = np.dot(data.T - int(subtract_mean)* self.mean, self.modes) / np.sqrt(self.eigval) #nsam, neig

        return xi


    def eval(self, xi=None, neig=None, add_mean=True):
        """Evaluate the expansion at given latent features, up to a given number of eigenvalues.

        Args:
            xi (np.ndarray, optional): Array of latent features of size `(M,K)`. Default is None, which takes the pre-build latent features.
            neig (int, optional): Number of requested eigenvalues in the evaluation. Defaults to None, which takes all eigenvalues.
            add_mean (bool, optional): Whether to add the mean before evaluation or not. Defaults to True. Useful without addition, e.g. if we are evaluating perturbations or standard deviations.

        Returns:
            np.ndarray: Reduced-dimensional data array of size `(N,M)`.
        """
        assert(self.built)

        nx = self.modes.shape[0]

        if xi is None:
            xi = self.xi

        if neig is None:
            neig = nx

        data_red = int(add_mean)*self.mean + np.dot(xi[:, :neig] * np.sqrt(self.eigval[np.newaxis, :neig]), self.modes[:, :neig].T)
        data_red = data_red.T # now nx, nsam

        return data_red

    def eigval_clip(self):
        """Clip low or numerically negative eigenvalues."""
        self.eigval[self.eigval<1.e-14] = 1.e-14

    def compute_relvar(self):
        r"""Compute relative cumulative variances

        Returns:
            np.ndarray: An array of size `(N,K)` for cumulative variance fractions.
        """
        assert(self.built)

        tmp = self.modes * np.sqrt(self.eigval)
        rel_var = (np.cumsum(tmp * tmp, axis=1) + 0.0) / (np.sum(tmp*tmp, axis=1).reshape(-1, 1) + 0.0) #nx, neig
        return rel_var

    def plot_modes(self, imodes=None):
        """Plot eigenmodes.

        Args:
            imodes (int, optional): The number of requested modes. Defaults to None, which plots all of them.
        """
        assert(self.built)

        nx = self.modes.shape[0]
        if imodes is None:
            imodes = range(self.get_neig(threshold=0.99))

        _ = plt.figure(figsize=(12,9))
        plt.plot(range(nx), self.mean, label='Mean')
        for imode in imodes:
            plt.plot(range(nx), self.modes[:, imode]) #, label='Mode '+str(imode+1))
        plt.gca().set_xlabel('x')
        plt.gca().set_ylabel('Modes')
        plt.legend()
        plt.savefig('modes.png')
        plt.close()


    def plot_eig(self):
        """Plots the eigenvalues on both linear and log scales."""
        assert(self.built)

        _ = plt.figure(figsize=(12,9))
        plt.plot(range(1, self.eigval.shape[0]+1),self.eigval, 'o-')
        plt.gca().set_xlabel('Number of Eigenvalues')
        plt.gca().set_ylabel('Eigenvalue')
        plt.savefig('eig.png')
        plt.gca().set_yscale('log')
        plt.savefig('eig_log.png')
        plt.close()

    def plot_expvar(self):
        """Plots the explained variance."""
        assert(self.built)

        explained_variance = np.cumsum(self.eigval)/np.sum(self.eigval)

        _ = plt.figure(figsize=(12,9))
        plt.plot(range(1, self.eigval.shape[0]+1),explained_variance, 'o-')
        plt.gca().set_xlabel('Number of Eigenvalues')
        plt.gca().set_ylabel('Explained  Variance')
        plt.savefig('exp_variance.png')
        plt.close()

    def plot_all(self):
        """Executes all plotting routines."""
        self.plot_eig()
        self.plot_expvar()
        self.plot_modes()

    def get_neig(self, threshold=0.99):
        """Computes the number of eigenvalues needed for a given threshold.

        Args:
            threshold (float, optional): Threshold value. Defaults to 0.99.

        Returns:
            int: Number of eigenvalues needed such that the explained variance fraction is above the given threshold.
        """
        assert(self.built)
        explained_variance = np.cumsum(self.eigval)/np.sum(self.eigval)
        neig = np.where(explained_variance>threshold)[0][0]+1

        print(f'Number of eigenvalues requested : {neig}')
        return neig

    def modes_clip(self, neig_clip=None):
        """Clips the modes to reduce potential storage cost.

        Args:
            neig_clip (int, optional): Number of eigenvalues requested for truncation.
        """
        if neig_clip is None:
            neig_clip = self.get_neig(threshold=0.999)

        self.modes = self.modes[:, :neig_clip]
