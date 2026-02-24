#!/usr/bin/env python
"""Test script for linear regression."""

import numpy as np
from pytuq.lreg.lreg import lsq


def test_lsq_fit():
    # Fit a known linear model: y = 2*x + 3
    np.random.seed(42)
    N = 50
    x = np.random.rand(N, 1)

    # Basis matrix: [1, x] (intercept + slope)
    Amat = np.column_stack([np.ones(N), x[:, 0]])
    y = 3.0 + 2.0 * x[:, 0]

    reg = lsq()
    reg.fita(Amat, y)

    assert(reg.fitted)
    assert(np.isclose(reg.cf[0], 3.0, atol=1.e-10))
    assert(np.isclose(reg.cf[1], 2.0, atol=1.e-10))


def test_lsq_predict():
    # Predictions should match the fitted model
    np.random.seed(42)
    N = 50
    x = np.random.rand(N, 1)

    Amat = np.column_stack([np.ones(N), x[:, 0]])
    y = 3.0 + 2.0 * x[:, 0]

    reg = lsq()
    reg.fita(Amat, y)

    # Predict on new points
    N_test = 20
    x_test = np.random.rand(N_test, 1)
    Amat_test = np.column_stack([np.ones(N_test), x_test[:, 0]])

    y_pred, y_var, y_cov = reg.predicta(Amat_test)
    y_expected = 3.0 + 2.0 * x_test[:, 0]

    assert(np.allclose(y_pred, y_expected, atol=1.e-10))


def test_lsq_overdetermined():
    # Overdetermined system: least-squares should find best fit
    np.random.seed(42)
    N = 100
    x = np.random.rand(N)
    y_true = 1.0 + 2.0 * x
    noise = 0.01 * np.random.randn(N)
    y = y_true + noise

    Amat = np.column_stack([np.ones(N), x])
    reg = lsq()
    reg.fita(Amat, y)

    assert(np.isclose(reg.cf[0], 1.0, atol=0.05))
    assert(np.isclose(reg.cf[1], 2.0, atol=0.05))


def test_lsq_multidim():
    # Multi-dimensional linear regression: y = 1 + 2*x1 + 3*x2
    np.random.seed(42)
    N = 100
    x = np.random.rand(N, 2)

    Amat = np.column_stack([np.ones(N), x])
    y = 1.0 + 2.0 * x[:, 0] + 3.0 * x[:, 1]

    reg = lsq()
    reg.fita(Amat, y)

    assert(np.isclose(reg.cf[0], 1.0, atol=1.e-10))
    assert(np.isclose(reg.cf[1], 2.0, atol=1.e-10))
    assert(np.isclose(reg.cf[2], 3.0, atol=1.e-10))


def test_lsq_used_indices():
    # All basis functions should be used
    np.random.seed(42)
    N = 50
    K = 4
    Amat = np.random.rand(N, K)
    y = Amat @ np.array([1., 2., 3., 4.])

    reg = lsq()
    reg.fita(Amat, y)

    assert(len(reg.used) == K)
    assert(np.array_equal(reg.used, np.arange(K)))


if __name__ == '__main__':
    test_lsq_fit()
    test_lsq_predict()
    test_lsq_overdetermined()
    test_lsq_multidim()
    test_lsq_used_indices()
