#!/usr/bin/env python
"""Test script for Gaussian Process regression."""

import numpy as np
from pytuq.fit.gp import gp, kernel_rbf, kernel_sin


def test_kernel_rbf_self():
    # RBF kernel of a point with itself should be 1.0
    x = np.array([1.0, 2.0, 3.0])
    assert(np.isclose(kernel_rbf(x, x, corlength=1.0), 1.0))


def test_kernel_rbf_symmetry():
    # Kernel should be symmetric: K(x, y) = K(y, x)
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    assert(np.isclose(kernel_rbf(x, y, 1.0), kernel_rbf(y, x, 1.0)))


def test_kernel_rbf_decay():
    # RBF kernel should decay with distance
    x = np.array([0.0])
    y_near = np.array([0.1])
    y_far = np.array([10.0])

    k_near = kernel_rbf(x, y_near, corlength=1.0)
    k_far = kernel_rbf(x, y_far, corlength=1.0)

    assert(k_near > k_far)


def test_kernel_rbf_corlength():
    # Larger correlation length should give higher kernel value for same distance
    x = np.array([0.0])
    y = np.array([1.0])

    k_short = kernel_rbf(x, y, corlength=0.5)
    k_long = kernel_rbf(x, y, corlength=5.0)

    assert(k_long > k_short)


def test_kernel_sin_self():
    # Sinusoidal kernel of a point with itself should be 1.0
    x = np.array([1.0, 2.0])
    assert(np.isclose(kernel_sin(x, x, corlength=1.0, period=1.0), 1.0))


def test_kernel_sin_symmetry():
    # Sinusoidal kernel should be symmetric
    x = np.array([1.0])
    y = np.array([2.0])
    assert(np.isclose(kernel_sin(x, y, 1.0, 2.0), kernel_sin(y, x, 1.0, 2.0)))


def test_gp_fit_predict_noiseless():
    # GP should interpolate noiseless training data exactly
    np.random.seed(42)
    N = 10
    x_train = np.linspace(0, 1, N).reshape(-1, 1)
    y_train = np.sin(2 * np.pi * x_train[:, 0])

    mygp = gp(kernel_rbf, [0.3], nugget=1.e-10)
    mygp.fit(x_train, y_train)

    assert(mygp.fitted)

    # Predict at training points: should recover training data
    y_pred, _, _ = mygp.predict(x_train, msc=0)
    assert(np.allclose(y_pred, y_train, atol=1.e-4))


def test_gp_predict_variance():
    # Variance at training points should be near zero; away from data should be larger
    np.random.seed(42)
    N = 10
    x_train = np.linspace(0, 1, N).reshape(-1, 1)
    y_train = np.sin(2 * np.pi * x_train[:, 0])

    mygp = gp(kernel_rbf, [0.3], nugget=1.e-10)
    mygp.fit(x_train, y_train)

    # Variance at training points
    _, y_var_train, _ = mygp.predict(x_train, msc=1)
    assert(np.all(y_var_train >= 0.0))

    # Variance at far-away points should be larger
    x_far = np.array([[5.0], [10.0]])
    _, y_var_far, _ = mygp.predict(x_far, msc=1)
    assert(np.all(y_var_far > np.mean(y_var_train)))


def test_gp_predict_covariance():
    # Full covariance mode: should return an (N,N) PSD matrix
    np.random.seed(42)
    N = 8
    x_train = np.linspace(0, 1, N).reshape(-1, 1)
    y_train = x_train[:, 0] ** 2

    mygp = gp(kernel_rbf, [0.5], nugget=1.e-10)
    mygp.fit(x_train, y_train)

    N_test = 5
    x_test = np.linspace(0.1, 0.9, N_test).reshape(-1, 1)
    _, y_var, y_cov = mygp.predict(x_test, msc=2)

    assert(y_cov.shape == (N_test, N_test))
    # Covariance matrix should be symmetric
    assert(np.allclose(y_cov, y_cov.T, atol=1.e-10))
    # Diagonal of covariance should match variance
    assert(np.allclose(np.diag(y_cov), y_var, atol=1.e-10))


def test_gp_posterior_predictive():
    # Posterior-predictive should add data variance to prediction variance
    np.random.seed(42)
    N = 8
    x_train = np.linspace(0, 1, N).reshape(-1, 1)
    y_train = x_train[:, 0]

    mygp = gp(kernel_rbf, [0.5], nugget=1.e-10)
    mygp.fit(x_train, y_train)

    x_test = np.array([[0.5]])
    _, y_var, _ = mygp.predict(x_test, msc=1, pp=False)
    _, y_var_pp, _ = mygp.predict(x_test, msc=1, pp=True)

    # Posterior-predictive variance should be larger
    assert(np.all(y_var_pp >= y_var))


def test_gp_with_basis():
    # GP with a linear basis should fit linear data well
    np.random.seed(42)
    N = 15
    x_train = np.linspace(0, 2, N).reshape(-1, 1)
    y_train = 3.0 + 2.0 * x_train[:, 0]

    def linear_basis(x, pars):
        return np.column_stack([np.ones(x.shape[0]), x[:, 0]])

    mygp = gp(kernel_rbf, [0.5], nugget=1.e-10, basis=(linear_basis, []))
    mygp.fit(x_train, y_train)

    assert(mygp.fitted)
    assert(mygp.basisEvaluatorSet)
    assert(mygp.nbas == 2)

    # Should interpolate
    y_pred, _, _ = mygp.predict(x_train, msc=0)
    assert(np.allclose(y_pred, y_train, atol=1.e-3))


def test_gp_multidim_input():
    # GP should work with multi-dimensional inputs
    np.random.seed(42)
    N = 20
    x_train = np.random.rand(N, 2)
    y_train = np.sum(x_train, axis=1)

    mygp = gp(kernel_rbf, [0.5], nugget=1.e-8)
    mygp.fit(x_train, y_train)

    y_pred, _, _ = mygp.predict(x_train, msc=0)
    assert(np.allclose(y_pred, y_train, atol=1.e-2))


def test_gp_kmat_self_shape():
    # Self kernel matrix should be (N, N) and symmetric positive semi-definite
    np.random.seed(42)
    N = 10
    x = np.random.rand(N, 2)

    mygp = gp(kernel_rbf, [1.0], nugget=1.e-10)
    mygp.kernel_params = [1.0]
    Kmat = mygp.get_kmat_self(x)

    assert(Kmat.shape == (N, N))
    assert(np.allclose(Kmat, Kmat.T))
    # Diagonal should be 1.0 (kernel_rbf of point with itself)
    assert(np.allclose(np.diag(Kmat), 1.0))
    # All eigenvalues should be non-negative
    eigvals = np.linalg.eigvalsh(Kmat)
    assert(np.all(eigvals > -1.e-10))


def test_gp_kmat_cross_shape():
    # Cross kernel matrix should be (N_test, N_train)
    np.random.seed(42)
    N_train = 10
    N_test = 5
    x_train = np.random.rand(N_train, 2)
    x_test = np.random.rand(N_test, 2)

    mygp = gp(kernel_rbf, [1.0], nugget=1.e-10)
    mygp.fit(x_train, np.random.rand(N_train))

    Kcross = mygp.get_kmat_cross(x_test)
    assert(Kcross.shape == (N_test, N_train))


def test_gp_predict_wstd():
    # predict_wstd should return mean and standard deviation
    np.random.seed(42)
    N = 10
    x_train = np.linspace(0, 1, N).reshape(-1, 1)
    y_train = np.sin(x_train[:, 0])

    mygp = gp(kernel_rbf, [0.3], nugget=1.e-10)
    mygp.fit(x_train, y_train)

    x_test = np.array([[0.5], [0.7]])
    y_mean, y_std = mygp.predict_wstd(x_test)

    assert(y_mean.shape == (2,))
    assert(y_std.shape == (2,))
    assert(np.all(y_std >= 0.0))


def test_gp_kernel_string():
    # GP should accept kernel as string 'RBF'
    np.random.seed(42)
    N = 10
    x_train = np.linspace(0, 1, N).reshape(-1, 1)
    y_train = x_train[:, 0] ** 2

    mygp = gp('RBF', [0.5], nugget=1.e-10)
    mygp.fit(x_train, y_train)

    assert(mygp.fitted)
    y_pred, _, _ = mygp.predict(x_train, msc=0)
    assert(np.allclose(y_pred, y_train, atol=1.e-3))


if __name__ == '__main__':
    test_kernel_rbf_self()
    test_kernel_rbf_symmetry()
    test_kernel_rbf_decay()
    test_kernel_rbf_corlength()
    test_kernel_sin_self()
    test_kernel_sin_symmetry()
    test_gp_fit_predict_noiseless()
    test_gp_predict_variance()
    test_gp_predict_covariance()
    test_gp_posterior_predictive()
    test_gp_with_basis()
    test_gp_multidim_input()
    test_gp_kmat_self_shape()
    test_gp_kmat_cross_shape()
    test_gp_predict_wstd()
    test_gp_kernel_string()
