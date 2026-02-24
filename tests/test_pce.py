#!/usr/bin/env python
"""Test script for PCE surrogate."""

import numpy as np
from pytuq.surrogates.pce import PCE


def test_pce_build_lsq():
    # Build a PCE surrogate using least-squares regression
    np.random.seed(42)
    dim = 2
    order = 3

    # Generate training data from a known polynomial
    N = 100
    x_train = np.random.rand(N, dim) * 2 - 1  # in [-1, 1]
    y_train = 1.0 + 2.0 * x_train[:, 0] + 3.0 * x_train[:, 1]

    pce = PCE(dim, order, 'LU', verbose=0)
    pce.set_training_data(x_train, y_train)
    cfs = pce.build(regression='lsq')

    assert(cfs is not None)


def test_pce_evaluate():
    # PCE should reproduce training data for polynomial targets
    np.random.seed(42)
    dim = 1
    order = 3

    N = 50
    x_train = np.random.rand(N, dim) * 2 - 1
    y_train = 1.0 + 2.0 * x_train[:, 0] + 0.5 * (3 * x_train[:, 0]**2 - 1) / 2

    pce = PCE(dim, order, 'LU', verbose=0)
    pce.set_training_data(x_train, y_train)
    pce.build(regression='lsq')

    result = pce.evaluate(x_train)
    y_pred = result['Y_eval']

    assert(np.allclose(y_pred, y_train, atol=1.e-8))


def test_pce_linear():
    # PCE for linear function should recover exact coefficients
    np.random.seed(42)
    dim = 2
    order = 1

    N = 50
    x_train = np.random.rand(N, dim) * 2 - 1
    y_train = 3.0 + 2.0 * x_train[:, 0] - 1.0 * x_train[:, 1]

    pce = PCE(dim, order, 'LU', verbose=0)
    pce.set_training_data(x_train, y_train)
    pce.build(regression='lsq')

    # Evaluate on new points
    N_test = 30
    x_test = np.random.rand(N_test, dim) * 2 - 1
    y_test = 3.0 + 2.0 * x_test[:, 0] - 1.0 * x_test[:, 1]

    result = pce.evaluate(x_test)
    y_pred = result['Y_eval']

    assert(np.allclose(y_pred, y_test, atol=1.e-8))


def test_pce_output_keys():
    # Evaluate should return dict with expected keys
    np.random.seed(42)
    dim = 1
    order = 2

    N = 30
    x_train = np.random.rand(N, dim) * 2 - 1
    y_train = x_train[:, 0]**2

    pce = PCE(dim, order, 'LU', verbose=0)
    pce.set_training_data(x_train, y_train)
    pce.build(regression='lsq')

    result = pce.evaluate(x_train)

    assert('Y_eval' in result)
    assert('Y_eval_std' in result)


def test_pce_pc_terms():
    # Number of PC terms should match expected
    dim = 3
    order = 2
    pce = PCE(dim, order, 'LU', verbose=0)

    terms = pce.get_pc_terms()

    # For total-order truncation: C(p+d, d) = C(5, 3) = 10
    assert(terms[0] == 10)


if __name__ == '__main__':
    test_pce_build_lsq()
    test_pce_evaluate()
    test_pce_linear()
    test_pce_output_keys()
    test_pce_pc_terms()
