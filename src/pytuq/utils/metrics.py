#!/usr/bin/env python
"""Module for various metrics of comparison."""

import numpy as np

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
    num = rmse(predictions, targets)/(np.max(targets)-np.min(targets))

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
