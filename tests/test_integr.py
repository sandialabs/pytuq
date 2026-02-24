#!/usr/bin/env python
"""Test script for Monte Carlo integration."""

import numpy as np
from pytuq.ftools.integr import IntegratorMC


def test_integr_constant():
    # Integral of constant function over [0,1]^d = constant * volume
    def const_func(x):
        return 3.0 * np.ones((x.shape[0], 1))

    domain = np.array([[0., 1.], [0., 1.]])

    integrator = IntegratorMC(seed=42)
    result, info = integrator.integrate(const_func, domain=domain, func_args={}, nmc=1000)

    assert(np.isclose(result, 3.0, atol=0.01))


def test_integr_linear():
    # Integral of f(x) = x over [0,1] = 0.5
    def linear_func(x):
        return x[:, 0:1]

    domain = np.array([[0., 1.]])

    integrator = IntegratorMC(seed=42)
    result, info = integrator.integrate(linear_func, domain=domain, func_args={}, nmc=50000)

    assert(np.isclose(result, 0.5, atol=0.01))


def test_integr_quadratic():
    # Integral of f(x) = x^2 over [0,1] = 1/3
    def quad_func(x):
        return x[:, 0:1]**2

    domain = np.array([[0., 1.]])

    integrator = IntegratorMC(seed=42)
    result, info = integrator.integrate(quad_func, domain=domain, func_args={}, nmc=50000)

    assert(np.isclose(result, 1.0 / 3.0, atol=0.01))


def test_integr_2d():
    # Integral of f(x,y) = x*y over [0,1]^2 = 0.25
    def prod_func(x):
        return (x[:, 0] * x[:, 1]).reshape(-1, 1)

    domain = np.array([[0., 1.], [0., 1.]])

    integrator = IntegratorMC(seed=42)
    result, info = integrator.integrate(prod_func, domain=domain, func_args={}, nmc=50000)

    assert(np.isclose(result, 0.25, atol=0.02))


def test_integr_neval():
    # Check that neval in results matches requested nmc
    def func(x):
        return np.ones((x.shape[0], 1))

    domain = np.array([[0., 1.]])
    nmc = 500

    integrator = IntegratorMC(seed=42)
    result, info = integrator.integrate(func, domain=domain, func_args={}, nmc=nmc)

    assert(info['neval'] == nmc)


if __name__ == '__main__':
    test_integr_constant()
    test_integr_linear()
    test_integr_quadratic()
    test_integr_2d()
    test_integr_neval()
