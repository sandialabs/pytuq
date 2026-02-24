#!/usr/bin/env python
"""Test script for function operations (arithmetic, composition, slicing)."""

import numpy as np
from pytuq.func import oper, toy


def test_add_fcn():
    # (f + g)(x) = f(x) + g(x)
    dim = 2
    f = toy.Exp(np.array([1.0, 0.0]))
    g = toy.Exp(np.array([0.0, 1.0]))
    h = f + g

    NN = 10
    xx = np.random.rand(NN, dim)

    yy = h(xx)
    expected = f(xx) + g(xx)
    assert(np.allclose(yy, expected))


def test_sub_fcn():
    # (f - g)(x) = f(x) - g(x)
    dim = 2
    f = toy.Exp(np.array([1.0, 0.0]))
    g = toy.Exp(np.array([0.0, 1.0]))
    h = f - g

    NN = 10
    xx = np.random.rand(NN, dim)

    yy = h(xx)
    expected = f(xx) - g(xx)
    assert(np.allclose(yy, expected))


def test_mult_fcn():
    # (f * g)(x) = f(x) * g(x)
    dim = 2
    f = toy.Exp(np.array([1.0, 0.0]))
    g = toy.Exp(np.array([0.0, 1.0]))
    h = f * g

    NN = 10
    xx = np.random.rand(NN, dim)

    yy = h(xx)
    expected = f(xx) * g(xx)
    assert(np.allclose(yy, expected))


def test_mult_grad():
    # Verify product rule gradient against numerical gradient
    dim = 2
    f = toy.Exp(np.array([1.0, 0.0]))
    g = toy.Exp(np.array([0.0, 1.0]))
    h = f * g

    NN = 10
    xx = np.random.rand(NN, dim)

    gg = h.grad(xx)
    gg_num = h.grad_(xx)

    assert(np.allclose(gg, gg_num, atol=1.e-8))


def test_div_fcn():
    # (f / g)(x) = f(x) / g(x)
    dim = 1
    f = toy.Exp(np.array([2.0]))
    g = toy.Exp(np.array([1.0]))
    h = f / g

    NN = 10
    xx = np.random.rand(NN, dim)

    yy = h(xx)
    expected = f(xx) / g(xx)
    assert(np.allclose(yy, expected))


def test_pow_fcn():
    # (f ** 2)(x) = f(x) ** 2
    dim = 1
    f = toy.Identity(dim)
    h = f ** 2

    NN = 10
    xx = np.random.rand(NN, dim)

    yy = h(xx)
    expected = xx ** 2
    assert(np.allclose(yy, expected))


def test_shift_fcn():
    # ShiftFcn(f, shift)(x) = f(x - shift)
    dim = 2
    shift = np.array([1.0, 2.0])
    f = toy.Exp(np.array([1.0, 1.0]))
    h = oper.ShiftFcn(f, shift)

    NN = 10
    xx = np.random.rand(NN, dim) + 3.0  # ensure x - shift > 0

    yy = h(xx)
    expected = f(xx - shift)
    assert(np.allclose(yy, expected))


def test_lintransform_fcn():
    # LinTransformFcn(f, scale, shift)(x) = scale * f(x) + shift
    dim = 1
    f = toy.Identity(dim)
    scale = 3.0
    shift = 5.0
    h = oper.LinTransformFcn(f, scale, shift)

    NN = 10
    xx = np.random.rand(NN, dim)

    yy = h(xx)
    expected = scale * xx + shift
    assert(np.allclose(yy, expected))


def test_pickdim():
    # PickDim picks a single dimension
    dim = 5
    pdim = 2
    cf = 3.0
    fcn = oper.PickDim(dim, pdim, cf=cf)

    NN = 10
    xx = np.random.rand(NN, dim)

    yy = fcn(xx)
    expected = (cf * xx[:, pdim]).reshape(-1, 1)
    assert(np.allclose(yy, expected))


def test_slice_fcn():
    # SliceFcn evaluates on a subset of dimensions
    dim = 3
    f = toy.Exp(np.array([1.0, 2.0, 3.0]))
    ind = [0, 2]
    nom = np.array([0.5, 0.5, 0.5])
    h = oper.SliceFcn(f, ind=ind, nom=nom)

    NN = 10
    xx = np.random.rand(NN, 2)

    yy = h(xx)
    # Reconstruct full input with nominal values
    xx_full = np.tile(nom, (NN, 1))
    xx_full[:, 0] = xx[:, 0]
    xx_full[:, 2] = xx[:, 1]
    expected = f(xx_full)
    assert(np.allclose(yy, expected))


def test_cartes_prod():
    # CartesProdFcn(f, g)(x, y) = f(x) * g(y)
    f = toy.Identity(1)
    g = toy.Identity(1)
    h = oper.CartesProdFcn(f, g)

    NN = 10
    xx = np.random.rand(NN, 2)

    yy = h(xx)
    expected = (xx[:, 0] * xx[:, 1]).reshape(-1, 1)
    assert(np.allclose(yy, expected))


if __name__ == '__main__':
    test_add_fcn()
    test_sub_fcn()
    test_mult_fcn()
    test_mult_grad()
    test_div_fcn()
    test_pow_fcn()
    test_shift_fcn()
    test_lintransform_fcn()
    test_pickdim()
    test_slice_fcn()
    test_cartes_prod()
