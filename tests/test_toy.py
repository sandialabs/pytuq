#!/usr/bin/env python
"""Test script for toy functions and their gradients."""

import numpy as np
from pytuq.func.toy import Constant, Identity, Exp, Log, Quad


def test_constant():
    # Constant function should return the same value everywhere
    const_val = np.array([3.14])
    fcn = Constant(dim=3, const=const_val)

    NN = 20
    xx = np.random.rand(NN, 3)

    yy = fcn(xx)
    assert(yy.shape == (NN, 1))
    assert(np.allclose(yy, const_val))


def test_identity():
    # Identity function should return the input
    dim = 4
    fcn = Identity(dim)

    NN = 15
    xx = np.random.rand(NN, dim)

    yy = fcn(xx)
    assert(yy.shape == (NN, dim))
    assert(np.allclose(yy, xx))


def test_identity_grad():
    # Gradient of identity is the identity matrix
    dim = 3
    fcn = Identity(dim)

    NN = 10
    xx = np.random.rand(NN, dim)

    gg = fcn.grad(xx)
    assert(gg.shape == (NN, dim, dim))
    for i in range(NN):
        assert(np.allclose(gg[i], np.eye(dim)))


def test_exp():
    # Exp function: f(x) = exp(w^T x)
    weights = np.array([1.0, -2.0])
    fcn = Exp(weights)

    NN = 10
    xx = np.random.rand(NN, 2)

    yy = fcn(xx)
    expected = np.exp(xx @ weights).reshape(-1, 1)
    assert(np.allclose(yy, expected))


def test_exp_grad():
    # Gradient of exp(w^T x) = w * exp(w^T x)
    weights = np.array([1.0, -2.0])
    fcn = Exp(weights)

    NN = 10
    xx = np.random.rand(NN, 2)

    gg = fcn.grad(xx)
    gg_num = fcn.grad_(xx)

    assert(np.allclose(gg, gg_num, atol=1.e-8))


def test_log():
    # Log function: f(x) = log(|w^T x|)
    weights = np.array([1.0])
    fcn = Log(weights)

    NN = 10
    xx = np.random.rand(NN, 1) + 0.1  # positive values

    yy = fcn(xx)
    expected = np.log(np.abs(xx @ weights)).reshape(-1, 1)
    assert(np.allclose(yy, expected))


def test_log_grad():
    # Verify gradient of log against numerical gradient
    weights = np.array([2.0])
    fcn = Log(weights)

    NN = 10
    xx = np.random.rand(NN, 1) + 0.5

    gg = fcn.grad(xx)
    gg_num = fcn.grad_(xx)

    assert(np.allclose(gg, gg_num, atol=1.e-8))


def test_quad():
    # Quad function: f(x) = 3 + x - x^2
    fcn = Quad()

    NN = 10
    xx = np.random.rand(NN, 1)

    yy = fcn(xx)
    expected = (3.0 + xx[:, 0] - xx[:, 0]**2).reshape(-1, 1)
    assert(np.allclose(yy, expected))


def test_quad_grad():
    # Gradient of Quad: f'(x) = 1 - 2x
    fcn = Quad()

    NN = 10
    xx = np.random.rand(NN, 1)

    gg = fcn.grad(xx)
    gg_num = fcn.grad_(xx)

    assert(np.allclose(gg, gg_num, atol=1.e-8))


if __name__ == '__main__':
    test_constant()
    test_identity()
    test_identity_grad()
    test_exp()
    test_exp_grad()
    test_log()
    test_log_grad()
    test_quad()
    test_quad_grad()
