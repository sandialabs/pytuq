#!/usr/bin/env python
"""Collection of various test functions."""

import numpy as np

def sin4(xx):
    r"""Fourth order of a shifted sine:

    .. math::
        f(x_1, ..., x_d) = \sin(2\sum_{i=1}^d x_i - 0.3)^4.

    Args:
        xx (np.ndarray): 2d input array :math:`x` of size `(N,d)`.

    Returns:
        np.ndarray: 1d array of size `N`.
    """
    assert(len(xx.shape)==2)

    y = np.sin(2.*np.sum(xx, axis=1)-0.3)**4
    return y

def const(xx):
    r"""Constant function:

    .. math::
        f(x_1,...,x_d)=3.

    Args:
        xx (np.ndarray): 2d input array :math:`x` of size `(N,d)` or 1d array of size `N`.

    Returns:
        np.ndarray: 1d array of size `N`.
    """

    y = 3.*np.ones(xx.shape[0])

    return y


def f2d(xx):
    r"""A simple bivariate function:

    .. math::
        f(x_1,x_2)=-x_1+x_2^2.

    Args:
        xx (np.ndarray): 2d input array :math:`x` of size `(N,2)`.

    Returns:
        np.ndarray: 1d array of size `N`.
    """
    assert(len(xx.shape)==2)
    assert(xx.shape[1]==2)

    y = -xx[:,0]+xx[:,1]**2

    return y

def cosine(xx):
    r"""Cosine function:

    .. math::
        f(x_1,...,x_d)=\cos(x_1).

    Args:
        xx (np.ndarray): 2d input array :math:`x` of size `(N,d)`.

    Returns:
        np.ndarray: 1d array of size `N`.
    """
    assert(len(xx.shape)==2)

    y = np.cos(xx[:,0])

    return y

def sinsum(xx):
    r"""Sine of sum function:

    .. math::
        f(x_1, ..., x_d) = \sin(\sum_{i=1}^d x_i).

    Args:
        xx (np.ndarray): 2d input array :math:`x` of size `(N,d)`.

    Returns:
        np.ndarray: 1d array of size `N`.
    """
    assert(len(xx.shape)==2)

    y = np.sin(np.sum(xx, axis=1))

    return y

def prodabs(xx):
    r"""Absolute product function:

    .. math::
        f(x_1, ..., x_d) = \prod_{i=1}^d |x_i|).

    Args:
        xx (np.ndarray): 2d input array :math:`x` of size `(N,d)`.

    Returns:
        np.ndarray: 1d array of size `N`.
    """
    assert(len(xx.shape)==2)

    y = np.abs(np.prod(xx, axis=1))

    return y
