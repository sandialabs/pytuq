#!/usr/bin/env python
"""Test script for Genz test functions and integration."""

import numpy as np
from pytuq.func.genz import GenzOscillatory, GenzSum, GenzCornerPeak


def test_genz_oscillatory_eval():
    # GenzOscillatory: f(x) = cos(2*pi*shift + w^T x)
    weights = np.array([1.0, 2.0])
    shift = 0.5
    fcn = GenzOscillatory(shift=shift, weights=weights)

    NN = 20
    xx = np.random.rand(NN, 2)

    yy = fcn(xx)
    expected = np.cos(2 * np.pi * shift + xx @ weights).reshape(-1, 1)
    assert(np.allclose(yy, expected))


def test_genz_oscillatory_grad():
    # Verify gradient against numerical gradient
    weights = np.array([1.0, 2.0])
    fcn = GenzOscillatory(shift=0.3, weights=weights)

    NN = 10
    xx = np.random.rand(NN, 2)

    gg = fcn.grad(xx)
    gg_num = fcn.grad_(xx)

    assert(np.allclose(gg, gg_num, atol=1.e-8))


def test_genz_oscillatory_integral():
    # The GenzOscillatory has a known analytical integral over [0,1]^d
    weights = np.array([1.0, 2.0])
    shift = 0.0
    fcn = GenzOscillatory(shift=shift, weights=weights)

    # Analytical integral
    exact_integral = fcn.intgl()

    # Numerical integral via Monte Carlo (large sample)
    np.random.seed(42)
    nmc = 200000
    xx = np.random.rand(nmc, 2)
    yy = fcn(xx)
    mc_integral = np.mean(yy)

    assert(np.isclose(exact_integral, mc_integral, atol=0.01))


def test_genz_sum_eval():
    # GenzSum: f(x) = shift + w^T x
    weights = np.array([1.0, 3.0])
    shift = 2.0
    fcn = GenzSum(shift=shift, weights=weights)

    NN = 10
    xx = np.random.rand(NN, 2)

    yy = fcn(xx)
    expected = (shift + xx @ weights).reshape(-1, 1)
    assert(np.allclose(yy, expected))


def test_genz_sum_grad():
    # Verify gradient of GenzSum against numerical gradient
    weights = np.array([1.0, 3.0])
    fcn = GenzSum(shift=2.0, weights=weights)

    NN = 10
    xx = np.random.rand(NN, 2)

    gg = fcn.grad(xx)
    gg_num = fcn.grad_(xx)

    assert(np.allclose(gg, gg_num, atol=1.e-8))


def test_genz_cornerpeak_grad():
    # Verify gradient of GenzCornerPeak against numerical gradient
    weights = np.array([1.0, 2.0])
    fcn = GenzCornerPeak(weights=weights)

    NN = 10
    xx = np.random.rand(NN, 2) * 0.5  # stay away from singularity

    gg = fcn.grad(xx)
    gg_num = fcn.grad_(xx)

    assert(np.allclose(gg, gg_num, atol=1.e-6))


if __name__ == '__main__':
    test_genz_oscillatory_eval()
    test_genz_oscillatory_grad()
    test_genz_oscillatory_integral()
    test_genz_sum_eval()
    test_genz_sum_grad()
    test_genz_cornerpeak_grad()
