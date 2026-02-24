#!/usr/bin/env python
"""Tests for the minf (model inference) module."""

import numpy as np


# ── MCMC base and samplers ──────────────────────────────────────────────


def _gauss_logpost(x, mu=None, cov_inv=None):
    """Simple Gaussian log-posterior for testing MCMC."""
    if mu is None:
        mu = np.zeros_like(x)
    if cov_inv is None:
        cov_inv = np.eye(len(x))
    d = x - mu
    return -0.5 * d @ cov_inv @ d


def _gauss_logpost_grad(x, mu=None, cov_inv=None):
    """Gradient of the Gaussian log-posterior."""
    if mu is None:
        mu = np.zeros_like(x)
    if cov_inv is None:
        cov_inv = np.eye(len(x))
    return -cov_inv @ (x - mu)


def test_mcmcbase_setlogpost():
    """MCMCBase.setLogPost stores the callable."""
    from pytuq.minf.calib import MCMCBase

    mc = MCMCBase()
    assert mc.logPost is None
    mc.setLogPost(_gauss_logpost, None)
    assert mc.logPost is _gauss_logpost
    assert mc.logPostGrad is None


def test_amcmc_1d_gaussian():
    """AMCMC samples a 1-d Gaussian; chain mean ≈ true mean."""
    from pytuq.minf.mcmc import AMCMC

    np.random.seed(42)
    mu = np.array([3.0])
    cov_inv = np.array([[4.0]])  # variance = 0.25

    sampler = AMCMC(gamma=0.5, t0=50, tadapt=200)
    sampler.setLogPost(lambda x: _gauss_logpost(x, mu, cov_inv), None)
    res = sampler.run(5000, np.array([0.0]))

    chain = res['chain']
    assert chain.shape == (5001, 1)  # nmcmc+1 rows
    assert res['accrate'] > 0.1
    assert np.abs(chain[1000:].mean() - 3.0) < 0.3


def test_amcmc_2d_gaussian():
    """AMCMC samples a 2-d uncorrelated Gaussian."""
    from pytuq.minf.mcmc import AMCMC

    np.random.seed(123)
    mu = np.array([1.0, -2.0])
    cov_inv = np.diag([1.0, 4.0])  # var = [1, 0.25]

    sampler = AMCMC(gamma=0.3, t0=50, tadapt=200)
    sampler.setLogPost(lambda x: _gauss_logpost(x, mu, cov_inv), None)
    res = sampler.run(6000, np.array([0.0, 0.0]))

    chain = res['chain'][2000:]
    assert np.abs(chain[:, 0].mean() - 1.0) < 0.5
    assert np.abs(chain[:, 1].mean() + 2.0) < 0.5


def test_amcmc_result_keys():
    """AMCMC run returns all expected keys."""
    from pytuq.minf.mcmc import AMCMC

    np.random.seed(0)
    sampler = AMCMC()
    sampler.setLogPost(lambda x: _gauss_logpost(x), None)
    res = sampler.run(100, np.array([0.0]))

    for key in ('chain', 'mapparams', 'maxpost', 'accrate', 'logpost', 'alphas'):
        assert key in res, f"Missing key {key}"


def test_amcmc_map_params():
    """MAP parameters should be near the mode of the posterior."""
    from pytuq.minf.mcmc import AMCMC

    np.random.seed(7)
    mu = np.array([2.0])
    cov_inv = np.array([[10.0]])  # tight variance = 0.1

    sampler = AMCMC(gamma=0.3)
    sampler.setLogPost(lambda x: _gauss_logpost(x, mu, cov_inv), None)
    res = sampler.run(4000, np.array([1.5]))

    assert np.abs(res['mapparams'][0] - 2.0) < 0.5


def test_hmc_1d_gaussian():
    """HMC samples a 1-d Gaussian using leapfrog integrator."""
    from pytuq.minf.mcmc import HMC

    np.random.seed(42)
    mu = np.array([3.0])
    cov_inv = np.array([[4.0]])

    sampler = HMC(epsilon=0.1, L=10)
    sampler.setLogPost(
        lambda x: _gauss_logpost(x, mu, cov_inv),
        lambda x: _gauss_logpost_grad(x, mu, cov_inv),
    )
    res = sampler.run(3000, np.array([0.0]))

    chain = res['chain'][500:]
    assert np.abs(chain.mean() - 3.0) < 0.3
    assert res['accrate'] > 0.3


def test_mala_1d_gaussian():
    """MALA samples a 1-d Gaussian."""
    from pytuq.minf.mcmc import MALA

    np.random.seed(42)
    mu = np.array([3.0])
    cov_inv = np.array([[4.0]])

    sampler = MALA(epsilon=0.1)
    sampler.setLogPost(
        lambda x: _gauss_logpost(x, mu, cov_inv),
        lambda x: _gauss_logpost_grad(x, mu, cov_inv),
    )
    res = sampler.run(3000, np.array([0.0]))

    chain = res['chain'][500:]
    assert np.abs(chain.mean() - 3.0) < 0.5
    assert res['accrate'] > 0.1


def test_logpost_values_stored():
    """Log-posterior values are stored along the chain."""
    from pytuq.minf.mcmc import AMCMC

    np.random.seed(0)
    sampler = AMCMC()
    sampler.setLogPost(lambda x: _gauss_logpost(x), None)
    res = sampler.run(200, np.array([0.5]))

    assert res['logpost'].shape == (201,)
    # first logpost should equal evaluation at the initial point
    assert np.isclose(res['logpost'][0], _gauss_logpost(np.array([0.5])))


# ── Infer data setup ────────────────────────────────────────────────────


def test_infer_setdata_1d():
    """Infer.setData with a 1-d array."""
    from pytuq.minf.infer import Infer

    inf = Infer(verbose=0)
    ydata = np.array([1.0, 2.0, 3.0])
    inf.setData(ydata)

    assert inf.ndata == 3
    assert inf.data_is_set


def test_infer_setdata_2d():
    """Infer.setData with a 2-d array (N x e)."""
    from pytuq.minf.infer import Infer

    inf = Infer(verbose=0)
    ydata = np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
    inf.setData(ydata)

    assert inf.ndata == 3
    assert len(inf.ydata[0]) == 2


def test_infer_setdata_list():
    """Infer.setData with a list of arrays (varying lengths)."""
    from pytuq.minf.infer import Infer

    inf = Infer(verbose=0)
    ydata = [[1.0, 1.1, 1.2], [2.0, 2.1]]
    inf.setData(ydata)

    assert inf.ndata == 2
    assert inf.neachs == [3, 2]


def test_infer_getdatastats():
    """Infer.getDataStats returns correct mean and variance."""
    from pytuq.minf.infer import Infer

    inf = Infer(verbose=0)
    ydata = [[1.0, 3.0], [10.0, 10.0]]
    inf.setData(ydata)

    mean, var = inf.getDataStats()
    assert np.isclose(mean[0], 2.0)
    assert np.isclose(mean[1], 10.0)
    assert np.isclose(var[0], 1.0)
    assert np.isclose(var[1], 0.0)


def test_infer_setdatavar_fixed():
    """Infer.setDataVar with var_fixed sets datavar correctly."""
    from pytuq.minf.infer import Infer

    inf = Infer(verbose=0)
    inf.setData(np.array([1.0, 2.0, 3.0]))
    inf.setDataVar('var_fixed', [0.01])

    assert inf.datavar_is_set
    assert np.allclose(inf.datavar, 0.01 * np.ones(3))
    assert inf.extrainferparams == 0


def test_infer_setdatavar_stdinfer():
    """Infer.setDataVar with std_infer adds an extra chain parameter."""
    from pytuq.minf.infer import Infer

    inf = Infer(verbose=0)
    inf.setData(np.array([1.0, 2.0]))
    inf.setDataVar('std_infer', None)

    assert inf.datavar_is_set
    assert inf.extrainferparams == 1


def test_infer_setdatavar_stdprop():
    """Infer.setDataVar with std_prop_fixed computes datavar from data means."""
    from pytuq.minf.infer import Infer

    inf = Infer(verbose=0)
    inf.setData(np.array([4.0, 8.0]))
    inf.setDataVar('std_prop_fixed', [0.1])  # stdfactor=0.1

    assert inf.datavar_is_set
    # std = 0.1 * |mean|, var = std^2
    assert np.isclose(inf.datavar[0], (0.1 * 4.0)**2)
    assert np.isclose(inf.datavar[1], (0.1 * 8.0)**2)


# ── MFVI ─────────────────────────────────────────────────────────────────


def test_mfvi_init():
    """MFVI initializes correctly with default priors."""
    from pytuq.minf.vi import MFVI

    model = lambda p: p @ np.array([[1.0, 2.0]])  # linear model
    y_data = np.array([1.0, 2.0])
    lossinfo = {'nmc': 50, 'datasigma': 0.1}

    vi = MFVI(model, y_data, pdim=2, lossinfo=lossinfo)
    assert vi.pdim == 2
    assert vi.nmc == 50
    assert len(vi.priors) == 2


def test_mfvi_get_posteriors():
    """MFVI.get_posteriors returns correct number of posterior distributions."""
    from pytuq.minf.vi import MFVI

    model = lambda p: p @ np.array([[1.0]])
    y_data = np.array([1.0])
    lossinfo = {'nmc': 10, 'datasigma': 0.1}

    vi = MFVI(model, y_data, pdim=1, lossinfo=lossinfo)
    var_params = np.array([0.5, 1.0])  # mean=0.5, sigma=1.0
    posts = vi.get_posteriors(var_params)

    assert len(posts) == 1
    assert np.isclose(posts[0].mu, 0.5)


def test_mfvi_loss_finite():
    """MFVI.eval_loss_elbo returns finite loss."""
    from pytuq.minf.vi import MFVI

    np.random.seed(42)
    model = lambda p: p @ np.array([[1.0, 0.5], [0.0, 1.0]])
    y_data = np.array([2.0, 1.0])
    lossinfo = {'nmc': 100, 'datasigma': 0.5}

    vi = MFVI(model, y_data, pdim=2, lossinfo=lossinfo)
    var_params = np.array([1.0, 0.5, 0.1, 0.1])  # means + sigmas
    loss = vi.eval_loss_elbo(var_params)

    assert np.isfinite(loss)


def test_mfvi_sample_posterior():
    """MFVI.sample_posterior returns correct shape."""
    from pytuq.minf.vi import MFVI

    np.random.seed(42)
    model = lambda p: p @ np.array([[1.0]])
    y_data = np.array([1.0])
    lossinfo = {'nmc': 10, 'datasigma': 0.1}

    vi = MFVI(model, y_data, pdim=3, lossinfo=lossinfo)
    var_params = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    samples = vi.sample_posterior(var_params, 50)

    assert samples.shape == (50, 3)


def test_mfvi_sigma_reparams():
    """MFVI reparameterization modes produce positive sigmas."""
    from pytuq.minf.vi import MFVI

    model = lambda p: p @ np.array([[1.0]])
    y_data = np.array([1.0])
    lossinfo = {'nmc': 10, 'datasigma': 0.1}

    for reparam in ['idt', 'exp', 'logexp']:
        vi = MFVI(model, y_data, pdim=1, lossinfo=lossinfo, reparam=reparam)
        # For exp and logexp, sigma(1.0) should be positive
        assert vi.sigma(1.0) > 0


def test_mfvi_loss_gmarg():
    """MFVI.eval_loss_gmarg returns finite loss."""
    from pytuq.minf.vi import MFVI

    np.random.seed(0)
    model = lambda p: p @ np.array([[1.0]])
    y_data = np.array([2.0])
    lossinfo = {'nmc': 50, 'datasigma': 1.0}

    vi = MFVI(model, y_data, pdim=1, lossinfo=lossinfo)
    var_params = np.array([2.0, 0.5])
    loss = vi.eval_loss_gmarg(var_params)

    assert np.isfinite(loss)


# ── Full inference pipeline (integration-level) ─────────────────────────


def test_model_infer_simple():
    """model_infer runs a simple 1-parameter fit end-to-end."""
    from pytuq.minf.minf import model_infer

    np.random.seed(42)

    # Linear model: f(p, q) = p[:, 0:1] * q[0]
    # q = [2.0], true parameter = 3.0, so model output = 6.0
    # 1 data location with 3 replicate observations
    def model(p, q):
        return (p[:, 0:1] * q[0])

    ydata = [[6.0, 5.8, 6.2]]  # 1 location, 3 replicates
    model_params = [2.0]
    domain = np.array([[0.0, 10.0]])

    results = model_infer(
        ydata, model, model_params, model_pdim=1,
        pr_type='uniform', pr_params={'domain': domain},
        dv_type='var_fixed', dv_params=[0.1],
        calib_type='amcmc',
        calib_params={'nmcmc': 2000, 'param_ini': None, 'gamma': 0.3},
        zflag=False,
    )

    assert 'chain' in results
    assert 'mapparams' in results
    chain = results['chain']
    assert chain.shape[0] == 2001
    # MAP should be near 3.0
    assert np.abs(results['mapparams'][0] - 3.0) < 1.5


# ── Likelihood classes ──────────────────────────────────────────────────


def test_likelihood_dummy():
    """Likelihood_dummy always returns 0.0."""
    from pytuq.minf.infer import Infer
    from pytuq.minf.likelihoods import Likelihood_dummy

    inf = Infer(verbose=0)
    inf.setData(np.array([1.0, 2.0]))
    inf.setDataVar('var_fixed', [0.01])

    lik = Likelihood_dummy(inf)
    assert lik.eval(np.array([0.5])) == 0.0
    assert lik.eval(np.array([100.0])) == 0.0


if __name__ == '__main__':
    test_mcmcbase_setlogpost()
    test_amcmc_1d_gaussian()
    test_amcmc_2d_gaussian()
    test_amcmc_result_keys()
    test_amcmc_map_params()
    test_hmc_1d_gaussian()
    test_mala_1d_gaussian()
    test_logpost_values_stored()
    test_infer_setdata_1d()
    test_infer_setdata_2d()
    test_infer_setdata_list()
    test_infer_getdatastats()
    test_infer_setdatavar_fixed()
    test_infer_setdatavar_stdinfer()
    test_infer_setdatavar_stdprop()
    test_mfvi_init()
    test_mfvi_get_posteriors()
    test_mfvi_loss_finite()
    test_mfvi_sample_posterior()
    test_mfvi_sigma_reparams()
    test_mfvi_loss_gmarg()
    test_model_infer_simple()
    test_likelihood_dummy()
