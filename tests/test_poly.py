#!/usr/bin/env python
"""Test script for polynomial functions."""

import numpy as np
from pytuq.func.poly import Leg, Mon
from pytuq.utils.mindex import get_mi


def test_legendre_constant():
    # Legendre polynomial with mindex=[[0]] and cfs=[c] should be constant
    mindex = np.array([[0]])
    cfs = np.array([3.14])
    fcn = Leg(mindex=mindex, cfs=cfs)

    NN = 20
    xx = np.random.rand(NN, 1) * 2 - 1  # in [-1, 1]

    yy = fcn(xx)
    assert(np.allclose(yy, 3.14))


def test_legendre_linear():
    # Legendre with mindex=[[0],[1]] and cfs=[a, b]
    # P_0(x) = 1, P_1(x) = x => f(x) = a + b*x
    mindex = np.array([[0], [1]])
    cfs = np.array([2.0, 3.0])
    fcn = Leg(mindex=mindex, cfs=cfs)

    NN = 20
    xx = np.random.rand(NN, 1) * 2 - 1

    yy = fcn(xx)
    expected = (2.0 + 3.0 * xx[:, 0]).reshape(-1, 1)
    assert(np.allclose(yy, expected))


def test_legendre_grad():
    # Verify gradient against numerical gradient
    mindex = get_mi(3, 2)
    np.random.seed(42)
    cfs = np.random.randn(mindex.shape[0])
    domain = np.array([[-1., 1.], [-1., 1.]])
    fcn = Leg(mindex=mindex, cfs=cfs, domain=domain)

    NN = 10
    xx = np.random.rand(NN, 2) * 2 - 1

    gg = fcn.grad(xx)
    gg_num = fcn.grad_(xx)

    assert(np.allclose(gg, gg_num, atol=1.e-6))


def test_monomial_eval():
    # Monomial with mindex=[[0],[1],[2]] and cfs=[1,0,1]
    # f(x) = 1 + x^2
    mindex = np.array([[0], [1], [2]])
    cfs = np.array([1.0, 0.0, 1.0])
    fcn = Mon(mindex=mindex, cfs=cfs)

    NN = 10
    xx = np.random.rand(NN, 1) * 2 - 1

    yy = fcn(xx)
    expected = (1.0 + xx[:, 0]**2).reshape(-1, 1)
    assert(np.allclose(yy, expected))


def test_monomial_grad():
    # Verify gradient against numerical gradient
    mindex = get_mi(3, 2)
    np.random.seed(42)
    cfs = np.random.randn(mindex.shape[0])
    fcn = Mon(mindex=mindex, cfs=cfs)

    NN = 10
    xx = np.random.rand(NN, 2)

    gg = fcn.grad(xx)
    gg_num = fcn.grad_(xx)

    assert(np.allclose(gg, gg_num, atol=1.e-6))


def test_poly_2d():
    # Test 2d polynomial evaluation shape
    mindex = get_mi(2, 3)
    np.random.seed(42)
    cfs = np.random.randn(mindex.shape[0])
    fcn = Leg(mindex=mindex, cfs=cfs)

    NN = 15
    xx = np.random.rand(NN, 3) * 2 - 1

    yy = fcn(xx)
    assert(yy.shape == (NN, 1))


if __name__ == '__main__':
    test_legendre_constant()
    test_legendre_linear()
    test_legendre_grad()
    test_monomial_eval()
    test_monomial_grad()
    test_poly_2d()
