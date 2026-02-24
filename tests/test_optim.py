#!/usr/bin/env python
"""Test script for optimization modules."""

import numpy as np
from pytuq.optim.optim import OptBase
from pytuq.optim.gd import GD, Adam
from pytuq.optim.sciwrap import ScipyWrapper


# ===== Objective functions for testing =====

def quadratic(x):
    """f(x) = sum(x^2), minimum at origin."""
    return np.sum(x ** 2)


def quadratic_grad(x):
    """Gradient of f(x) = sum(x^2)."""
    return 2.0 * x


def rosenbrock_2d(x):
    """Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1)."""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_2d_grad(x):
    """Gradient of 2d Rosenbrock."""
    dfdx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dfdy = 200 * (x[1] - x[0] ** 2)
    return np.array([dfdx, dfdy])


# ===== Tests for OptBase =====

def test_optbase_set_objective():
    # setObjective should store the callable
    opt = OptBase()
    opt.setObjective(quadratic, ObjectiveGrad=quadratic_grad)

    assert(opt.Objective is quadratic)
    assert(opt.ObjectiveGrad is quadratic_grad)
    assert(opt.ObjectiveHess is None)


def test_optbase_stepper_not_implemented():
    # Base class stepper should raise NotImplementedError
    opt = OptBase()
    try:
        opt.stepper(np.array([1.0]), 0)
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        pass


# ===== Tests for GD =====

def test_gd_quadratic():
    # GD should minimize a simple quadratic
    gd = GD(step_size=0.1)
    gd.setObjective(quadratic, ObjectiveGrad=quadratic_grad)

    x0 = np.array([5.0, -3.0])
    results = gd.run(200, x0)

    assert('best' in results)
    assert('bestobj' in results)
    assert('samples' in results)
    assert('objvalues' in results)

    # Should converge near the origin
    assert(np.allclose(results['best'], 0.0, atol=0.01))
    assert(results['bestobj'] < 0.001)


def test_gd_step_size_effect():
    # Smaller step size should converge more slowly
    gd_fast = GD(step_size=0.1)
    gd_fast.setObjective(quadratic, ObjectiveGrad=quadratic_grad)

    gd_slow = GD(step_size=0.01)
    gd_slow.setObjective(quadratic, ObjectiveGrad=quadratic_grad)

    x0 = np.array([5.0, -3.0])
    nsteps = 50

    results_fast = gd_fast.run(nsteps, x0)
    results_slow = gd_slow.run(nsteps, x0)

    # After same number of steps, faster step size should have lower objective
    assert(results_fast['bestobj'] < results_slow['bestobj'])


def test_gd_history_length():
    # History should have nsteps + 1 entries (initial + nsteps)
    gd = GD(step_size=0.1)
    gd.setObjective(quadratic, ObjectiveGrad=quadratic_grad)

    nsteps = 30
    x0 = np.array([1.0])
    results = gd.run(nsteps, x0)

    assert(results['samples'].shape[0] == nsteps + 1)
    assert(results['objvalues'].shape[0] == nsteps + 1)


def test_gd_decreasing_objective():
    # Objective should generally decrease for well-tuned GD on convex function
    gd = GD(step_size=0.1)
    gd.setObjective(quadratic, ObjectiveGrad=quadratic_grad)

    x0 = np.array([3.0, -2.0])
    results = gd.run(50, x0)

    # First value should be larger than last
    assert(results['objvalues'][0] > results['objvalues'][-1])


# ===== Tests for Adam =====

def test_adam_quadratic():
    # Adam should minimize a simple quadratic
    dim = 2
    adam = Adam(dim, lr=0.1)
    adam.setObjective(quadratic, ObjectiveGrad=quadratic_grad)

    x0 = np.array([5.0, -3.0])
    results = adam.run(300, x0)

    assert(np.allclose(results['best'], 0.0, atol=0.1))


def test_adam_rosenbrock():
    # Adam should make progress on Rosenbrock (may not fully converge)
    dim = 2
    adam = Adam(dim, lr=0.01)
    adam.setObjective(rosenbrock_2d, ObjectiveGrad=rosenbrock_2d_grad)

    x0 = np.array([0.0, 0.0])
    results = adam.run(1000, x0)

    # Should at least reduce from initial value
    assert(results['bestobj'] < rosenbrock_2d(x0))


# ===== Tests for ScipyWrapper =====

def test_scipy_bfgs_quadratic():
    # BFGS should minimize a quadratic exactly
    opt = ScipyWrapper(method='BFGS')
    opt.setObjective(quadratic, ObjectiveGrad=quadratic_grad)

    x0 = np.array([5.0, -3.0])
    results = opt.run(100, x0)

    assert(np.allclose(results['best'], 0.0, atol=1.e-6))
    assert(results['bestobj'] < 1.e-10)


def test_scipy_bfgs_rosenbrock():
    # BFGS should find the Rosenbrock minimum
    opt = ScipyWrapper(method='BFGS')
    opt.setObjective(rosenbrock_2d, ObjectiveGrad=rosenbrock_2d_grad)

    x0 = np.array([0.0, 0.0])
    results = opt.run(100, x0)

    assert(np.allclose(results['best'], [1.0, 1.0], atol=0.01))
    assert(results['bestobj'] < 0.001)


def test_scipy_lbfgsb_bounded():
    # L-BFGS-B should respect bounds
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    opt = ScipyWrapper(method='L-BFGS-B', bounds=bounds)
    opt.setObjective(quadratic, ObjectiveGrad=quadratic_grad)

    x0 = np.array([1.5, -1.5])
    results = opt.run(100, x0)

    assert(np.allclose(results['best'], 0.0, atol=1.e-6))
    # Best should be within bounds
    assert(np.all(results['best'] >= -2.0))
    assert(np.all(results['best'] <= 2.0))


def test_scipy_nelder_mead():
    # Nelder-Mead (gradient-free) should also find the minimum
    opt = ScipyWrapper(method='Nelder-Mead')
    opt.setObjective(quadratic)

    x0 = np.array([3.0, -2.0])
    results = opt.run(100, x0)

    assert(np.allclose(results['best'], 0.0, atol=0.01))


def test_scipy_results_keys():
    # Results dictionary should contain expected keys
    opt = ScipyWrapper(method='BFGS')
    opt.setObjective(quadratic, ObjectiveGrad=quadratic_grad)

    x0 = np.array([1.0])
    results = opt.run(100, x0)

    assert('best' in results)
    assert('bestobj' in results)
    assert('samples' in results)
    assert('objvalues' in results)


def test_scipy_with_objective_info():
    # Should handle extra ObjectiveInfo kwargs
    def weighted_quadratic(x, scale=1.0):
        return scale * np.sum(x ** 2)

    def weighted_quadratic_grad(x, p):
        return p['scale'] * 2.0 * x

    opt = ScipyWrapper(method='BFGS')
    opt.setObjective(weighted_quadratic, scale=2.0)

    x0 = np.array([3.0, -2.0])
    results = opt.run(100, x0)

    assert(np.allclose(results['best'], 0.0, atol=1.e-4))


if __name__ == '__main__':
    test_optbase_set_objective()
    test_optbase_stepper_not_implemented()
    test_gd_quadratic()
    test_gd_step_size_effect()
    test_gd_history_length()
    test_gd_decreasing_objective()
    test_adam_quadratic()
    test_adam_rosenbrock()
    test_scipy_bfgs_quadratic()
    test_scipy_bfgs_rosenbrock()
    test_scipy_lbfgsb_bounded()
    test_scipy_nelder_mead()
    test_scipy_results_keys()
    test_scipy_with_objective_info()
