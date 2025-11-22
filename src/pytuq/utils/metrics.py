#!/usr/bin/env python
"""Module for various metrics of comparison."""

import numpy as np
import scipy.stats as st

def rel_l2(predictions, targets):
    """Relative L2 metric.

    Args:
        predictions (np.ndarray): prediction array
        targets (np.ndarray): target array

    Returns:
        float: relative L2 error
    """
    # TODO: slowly remove this as it should be the same as rel_rmse()
    return np.linalg.norm(predictions - targets) / np.linalg.norm(targets)


def rmse(predictions, targets):
    """Root-mean-square error (RMSE).

    Args:
        predictions (np.ndarray): prediction array
        targets (np.ndarray): target array

    Returns:
        float: RMSE error
    """
    return np.sqrt(((predictions - targets) ** 2).mean())

def rel_rmse(predictions, targets):
    """Relative Root-mean-square error (RMSE).

    Args:
        predictions (np.ndarray): prediction array
        targets (np.ndarray): target array

    Returns:
        float: Relative RMSE error
    """
    return rmse(predictions, targets)/rmse(np.zeros_like(targets), targets)


def norm_rmse(predictions, targets):
    """Normalized root-mean-square error (RMSE).

    Args:
        predictions (np.ndarray): prediction array
        targets (np.ndarray): target array

    Returns:
        float: Normalized RMSE error
    """
    return rmse(predictions, targets)/(np.max(targets)-np.min(targets))

def fast_auc(y_prob, targets):
    """Fast Area-Under-Curve (AUC) computation.

    See: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013

    Args:
        targets (int np.ndarray): true array
        y_prob (float np.ndarray): predicted probabilities

    Returns:
        float: AUC metric
    """
    targets = np.asarray(targets)
    targets = targets[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(targets)
    for i in range(n):
        y_i = targets[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def mae(predictions, targets):
    """Mean absolute error (MAE).

    Args:
        predictions (np.ndarray): predicted array
        targets (np.ndarray): true array

    Returns:
        float: MAE metric
    """

    return np.abs(targets - predictions).mean()

def crps_gauss(y, mu, sig):
    r"""Continuous Ranked Probability Score (CRPS) for Gaussian forecast.

    Args:
        y (np.ndarray): true array
        mu (np.ndarray): mean array
        sig (np.ndarray): standard deviation array

    Returns:
        np.ndarray: CRPS metric
    """
    z = (y-mu)/sig
    rv = st.norm()
    crps = -sig*(1./np.sqrt(np.pi) - 2.*rv.pdf(z) - z*(2.*rv.cdf(z)-1.))
    return crps



def crps_samples(y, samples):
    r"""
    Compute CRPS from predictive samples.

    Args:
        y (np.ndarray): true array of shape `(M,)` or scalar
        samples (np.ndarray): predictive samples of shape `(M, N)` or `(N,)`.

    Returns:
        np.ndarray: CRPS metric of shape `(M,)`

    Notes:
        Empirical CRPS:
        CRPS = mean(|X - y|) - 0.5 * mean(|X_i - X_j|)
        where X_i are Monte Carlo samples from the predictive distribution.
    """
    samples = np.asarray(samples)
    y = np.asarray(y)

    if samples.ndim == 1:
        samples = samples[None, :]  # (1, N)
    if y.ndim == 0:
        y = np.array([y])

    assert samples.shape[0] == y.shape[0] or y.size == 1, \
        "y must be scalar or have same length as first dimension of samples"

    N = samples.shape[1]

    # |X - y|
    term1 = np.mean(np.abs(samples - y[:, None]), axis=1)

    # |X_i - X_j|
    # Efficient vectorized pairwise distance
    term2 = np.mean(np.abs(samples[:, :, None] - samples[:, None, :]), axis=(1, 2))

    return term1 - 0.5 * term2
