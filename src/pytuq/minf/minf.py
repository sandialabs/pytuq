#!/usr/bin/env python
"""Various utilities for model parameter inference and postprocessing."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from .infer import Infer
from .mcmc import AMCMC
from .calib import MCMCBase
from .likelihoods import *
from .priors import *

from ..utils.maps import scale01ToDom
from ..utils.plotting import lighten_color, plot_vars, plot_shade

def model_infer(ydata, model, model_params, model_pdim,
                inpdf_type='pci', pc_type='LU', outord=0,
                rndind=[], fixindnom=[], ind_calib=None,
                calib_type='amcmc', calib_params=None,
                md_transform=None, datamode=None,
                lik_type='classical', lik_params={},
                pr_type='uniform', pr_params={},
                dv_type='var_fixed', dv_params=None,
                zflag=True):
    """Main function that performs model parameter inference, with or without embedded model error.

    Args:
        ydata (list or np.ndarray): List of `N` 1d arrays corresponding to data for each design location, or a 2d array of size `(N,e)`, or an 1d array of size `N`.
        model (callable): Model with signature `f(p, q)`, where `p` are model parameters of interest, and `q` are other helpful model parameters.
        model_params (tuple or list): Model parameters `q`.
        model_pdim (int): Parameter dimensionality, i.e. number of model parameters.
        inpdf_type (str): Embedded PDF type. Options are 'pci' (default) and 'pct'.
        pc_type (str): Embedded PC type. Can be 'LU' (default) or 'HG'.
        outord (int, optional): Order for the output PC in the likelihood computation. Defaults to 0.
        rndind (list[ind], optional): List of indices of parameters to be embedded. If None, embeds in all parameters. Default is empty list, i.e. no embedding.
        fixindnom (list, optional): An array of size `(K,2)`, where first column indicates indices of parameters that are fixed (i.e. not part of the inference), and the second column is their nominal, fixed values. Defaults to an empty list, i.e. all parameters are inferred, none are fixed.
        ind_calib (list, optional): Model output indices that are used for calibration. Default is None, i.e. all outputs being used for calibration.
        calib_type (str or calib.Calib, optional): Calibration type. Only 'amcmc' (default) is implemented. Can also be a user-defined Calib object.
        calib_params (dict, optional): Dictionary of calibration parameters.
        md_transform (callable, optional): Potentially, a transform to be applied to model and data before the likelihood computation starts. Default is None, i.e. identity function.
        datamode (str, optional): If 'mean', work with data means per location.
        lik_type (str, optional): Likelihood type: options are 'classical', 'logclassical', 'abc', 'gausmarg', 'dummy'.
        lik_params (dict, optional): Parameters of likelihood.
        pr_type (str, optional): Prior type: the options are 'uniform' and 'normal'.
        pr_params (dict, optional): Parameters of prior.
        dv_type (str, optional): Data variance treatment type. Options are 'var_fixed', 'std_infer', 'std_infer_log', 'std_prop_fixed', 'var_fromdata_fixed', 'log_var_fromdata_fixed', 'var_fromdata_infer', 'log_var_fromdata_infer', 'scale_var'.
        dv_params (list, optional): Parameter list relevant to data variance computation. Defaults to None.
        zflag (bool, optional): Controls whether we want to precondition the inference with a deterministic optimization to find a better starting point for the chain.

    Returns:
        dict: Dictionary of results. Keys are 'chain' (chain samples array), 'mapparams' (MAP parameters array), 'maxpost' (maximal log-post value), 'accrate' (acceptance rate), 'logpost' (log-post values throughout the chain), 'alphas' (acceptance probabilities throughout the chain), 'post' (Inference object).
    """
    if lik_type=='classical' or lik_type=='log_classical':
        assert(len(rndind)==0)


    inference = Infer()

    inference.setData(ydata, datamode=datamode)
    inference.setDataVar(dv_type, dv_params)



    assert(model_pdim>=len(fixindnom))
    for fixind in fixindnom:
        assert(fixind[0]>=0)
        assert(fixind[0]<model_pdim)

    inference.setModelRVinput(inpdf_type, pc_type, model_pdim, rndind)
    inference.setModelRVoutput(outord)
    inference.setModel(model, model_params,
                       md_transform=md_transform,
                       fixindnom=fixindnom, ind_calib=ind_calib)


    # likelihoods = {'classical': Likelihood_classical(inference),
    #                'logclassical': Likelihood_logclassical(inference),
    #                'abc': Likelihood_abc(inference, **lik_params),
    #                'gausmarg': Likelihood_gausmarg(inference),
    #                'dummy': Likelihood_dummy(inference)}

    if lik_type == 'classical':
        inference.setLikelihood(Likelihood_classical(inference))
    elif lik_type == 'logclassical':
        inference.setLikelihood(Likelihood_logclassical(inference))
    elif lik_type == 'abc':
        inference.setLikelihood(Likelihood_abc(inference, **lik_params))
    elif lik_type == 'gausmarg':
        inference.setLikelihood(Likelihood_gausmarg(inference))
    elif lik_type == 'dummy':
        inference.setLikelihood(Likelihood_dummy(inference))
    else:
        print(f"Likelihood type {lik_type} unknown. Exiting.")
        sys.exit()

    if pr_type=='uniform':
        inference.setPrior(Prior_uniform(inference, **pr_params))
    elif pr_type=='normal':
        inference.setPrior(Prior_normal(inference, **pr_params))
    else:
        print(f"Prior type {pr_type} unknown. Exiting.")
        sys.exit()


    inference.setChain(default_init=0.01)

    if isinstance(calib_type, MCMCBase):
        calib = calib_type.copy()
    elif isinstance(calib_type, str):
        if calib_type == 'amcmc':
            if calib_params is None:
                calib_params = {}

            if calib_params['param_ini'] is None:
                param_ini = inference.chainInit
            else:
                param_ini = calib_params['param_ini']
                assert(param_ini.shape[0]==inference.chdim)

            if zflag:
                res = minimize((lambda x, fcn: -fcn(x)),
                               param_ini,
                               args=(inference.evalLogPost,),
                               method='BFGS', options={'gtol': 1e-13})
                print(res)
                print('Optimal values via BFGS:', res.x)
                param_ini = res.x



            nmcmc = calib_params['nmcmc']
            del calib_params['nmcmc']
            del calib_params['param_ini']

            calib = AMCMC(**calib_params)

        else:
            print(f'Calib_type {calib_type} not recognized. Exiting.')
            sys.exit()
    else:
        print(f'Calib_type type {type(calib_type)} not recognized. Exiting.')
        sys.exit()

    # Run the inference
    calib.setLogPost(inference.evalLogPost, None)
    calib_results = calib.run(nmcmc, param_ini)
    calib_results['post'] = inference


    return calib_results


def model_infer_postp(calib_results,
                      checkmode=None, nburn=0, nevery=1, nxi=1):
    """Postprocessing calibration results.

    Args:
        calib_results (dict): Result dictionary, the output of model_infer(...) function.
        checkmode (tuple, optional): Mode of evaluation a tuple of [model, model_parameters, index of calibrated outputs, final transform]. Defaults to None, which means get the inference object's internal model features.
        nburn (int, optional): Number of burned samples from the beginning of the chain. Default is 0, i.e. no burn-in.
        nevery (int, optional): Thinning of the chain. Default is 1, i.e. no thinning.
        nxi (int, optional): Number of samples per chain sample.

    Returns:
        tuple: ycheck (dictionary of various prediction means and variances), psamples (3d array of parameter samples of size Npost x nxi x pdim), fsamples (3d array of model output samples Npost x nxi x outdim)
    """
    inference = calib_results['post']
    nmcmc = len(calib_results['chain'])
    assert(nburn<nmcmc)
    ycheck = {}
    # Get MAP sample
    ycheck['map_mean'], ycheck['map_var'] = inference.getModelMoments_NISP(calib_results['mapparams'], fmode=checkmode)
    nmout = ycheck['map_mean'].shape[0]
    psample_map, fsample_map = inference.getIOSamples(calib_results['mapparams'], nxi=nxi, fmode=checkmode)
    nxi_, npar = psample_map.shape
    nxi__, nout = fsample_map.shape
    assert(nxi_==nxi and nxi__==nxi)
    assert(nmout==nout)

    npost = len(calib_results['chain'][nburn::nevery])
    psamples = np.empty((npost, nxi, npar))
    fsamples = np.empty((npost, nxi, nout))
    # Get MCMC prediction samples
    #xi = inference.inpcrv.sampleGerm(nxi)

    ycheck['mcmc_mean'] = np.empty((npost, nmout))
    ycheck['mcmc_var'] = np.empty((npost, nmout))

    chain_thinned = calib_results['chain'][nburn::nevery]
    for i in range(npost):
        sam = chain_thinned[i]
        psamples[i, :, :], fsamples[i, :, :] = inference.getIOSamples(sam, nxi=nxi, fmode=checkmode)
        ycheck['mcmc_mean'][i, :], ycheck['mcmc_var'][i, :] = inference.getModelMoments_NISP(sam, fmode=checkmode)
    ycheck['mean_mean'] = np.average(ycheck['mcmc_mean'], axis=0)
    ycheck['var_mean'] = np.var(ycheck['mcmc_mean'], axis=0) #Perr
    ycheck['mean_var'] = np.average(ycheck['mcmc_var'], axis=0) #Merr
    ycheck['var_var'] = np.var(ycheck['mcmc_var'], axis=0)

    np.savetxt('var_mean.txt',  ycheck['var_mean'])
    np.savetxt('mean_var.txt',  ycheck['mean_var'])

    samples = {}
    samples['pmcmc'], samples['fmcmc'] = psamples, fsamples
    samples['pmap'], samples['fmap'] = psample_map, fsample_map
    return ycheck, samples

def plot_1d_data(xdata, ydata, xygrid_true=None):
    """Plotting 1d data.

    Args:
        xdata (np.ndarray): An 1d array of x-data.
        ydata (np.ndarray): An 1d array of y-data.
        xygrid_true (tuple, optional): A tuple of gridded x- and y-values, if one wants to plot the true model as well.
    """
    plt.figure(figsize=(12, 9))
    npt = len(xdata)

    plt.plot(xdata, ydata, 'ko', zorder=1000, ms=11, label='Data, N = ' + str(npt))
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.xlim([np.min(xgrid), np.max(xgrid)])
    # ymin, ymax = np.min(ydata), np.max(ydata)
    # ymin, ymax = ymin - 0.25* (ymax-ymin), ymax + 0.25* (ymax-ymin)
    # plt.ylim([ymin, ymax])
    plt.legend(loc='upper left')
    plt.savefig('data_only_' + str(npt) + '.png')
    if xygrid_true is not None:
        xgrid, ygrid_true = xygrid_true
        plt.plot(xgrid, ygrid_true, 'k--', label='Truth')
        plt.savefig('data_model_' + str(npt) + '.png')
    plt.clf()

def plot_1d_samples(xdata, ydata, xygrid_samples=None):
    """Plotting 1d data.

    Args:
        xdata (np.ndarray): An 1d array of x-data.
        ydata (np.ndarray): An 1d array of y-data.
        xygrid_samples (tuple, optional): A tuple of gridded x- and y-values (can be a 2d array to plot many samples of the true model), if one wants to plot the true model as well.

    Note:
        This is very similar to plot_1d_data() and could be merged with it.
    """
    plt.figure(figsize=(12, 9))
    npt = len(xdata)


    plt.plot(xdata, ydata, 'ko', zorder=1000, ms=11, label='Data, N = ' + str(npt))
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.xlim([np.min(xgrid), np.max(xgrid)])
    # ymin, ymax = np.min(ydata), np.max(ydata)
    # ymin, ymax = ymin - 0.25* (ymax-ymin), ymax + 0.25* (ymax-ymin)
    # plt.ylim([ymin, ymax])
    plt.legend(loc='upper left')
    if xygrid_samples is not None:
        xgrid, ygrid_samples = xygrid_samples
        plt.plot(xgrid, ygrid_samples, 'r-', lw=1)
        plt.savefig('samples_model.png')
    plt.clf()


def plot_1dfit(xgrid, ygrid_pred, ygrid_true=None, xydata=None):
    """Plotting 1d fit with total and MAP standard deviation.

    Args:
        xgrid (np.ndarray): An 1d array of x-grid
        ygrid_pred (dict): Dictionary containing moments of predictions.
        ygrid_true (np.ndarray, optional): If not None, this is the true model values at the x-grid.
        xydata (tuple, optional): If not None, this is the x- and y-data tuple and plots on top of the fits.
    """
    plt.figure(figsize=(12,9))
    p, = plt.plot(xgrid, ygrid_pred['mean_mean'], 'b-', label='Mean')
    lc = lighten_color(p.get_color(), 0.5)
    plt.fill_between(xgrid,
                     ygrid_pred['mean_mean'] - np.sqrt(ygrid_pred['mean_var']+ygrid_pred['var_mean']),
                     ygrid_pred['mean_mean'] + np.sqrt(ygrid_pred['mean_var']+ygrid_pred['var_mean']),
                     color=lc, alpha=1.0, label='Total StDev')

    q, = plt.plot(xgrid, ygrid_pred['map_mean'], 'g-', label='MAP')
    lc = lighten_color(q.get_color(), 0.5)
    plt.fill_between(xgrid,
                     ygrid_pred['map_mean'] - np.sqrt(ygrid_pred['map_var']),
                     ygrid_pred['map_mean'] + np.sqrt(ygrid_pred['map_var']),
                     color=lc, alpha=0.5, label='MAP StDev')
    if ygrid_true is not None:
        plt.plot(xgrid, ygrid_true, 'k--', label='Truth')
    ymind, ymaxd = -np.inf, np.inf
    idata = 0
    if xydata is not None:
        idata = 1
        xdata, ydata = xydata
        plt.plot(xdata, ydata, 'ko', zorder=1000, ms=8,
                 markeredgecolor='w', label='Data')
        ymind, ymaxd = np.min(ydata), np.max(ydata)
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.xlim([np.min(xgrid), np.max(xgrid)])
    ymin, ymax = np.min(ygrid_pred['mean_mean']), np.max(ygrid_pred['mean_mean'])
    ymin, ymax = min(ymin, ymind), max(ymax, ymaxd)
    ymin, ymax = ymin - 0.15* (ymax-ymin), ymax + 0.15* (ymax-ymin)
    plt.ylim([ymin, ymax])
    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(h[:4+idata], l[:4+idata])

    plt.savefig('fit_1d.png')
    plt.clf()


def plot_1dfit_vars(xgrid, ygrid_pred, xydata=None):
    """Plotting 1d fit with variance decomposed into posterior uncertainty and model error.

    Args:
        xgrid (np.ndarray): An 1d array of x-grid
        ygrid_pred (dict): Dictionary containing moments of predictions.
        xydata (tuple, optional): If not None, this is the x- and y-data tuple and plots on top of the fits.
    """
    fig = plt.figure(figsize=(12, 9))
    thisax = plt.gca()
    variances = np.vstack((ygrid_pred['var_mean'], ygrid_pred['mean_var'])).T
    # np.savetxt('fvars.txt', variances)
    varlabels = ['Posterior uncertainty', 'Model error']
    varcolors = ['blue', 'lightblue']
    plot_vars(xgrid, ygrid_pred['mean_mean'], variances=variances, ysam=None,
                 stdfactor=1., varlabels=varlabels, varcolors=varcolors,
                 interp=None, connected=True, ax=thisax)

    ymind, ymaxd = -np.inf, np.inf
    idata = 0
    if xydata is not None:
        idata = 1
        xdata, ydata = xydata
        thisax.plot(xdata, ydata, 'ko', zorder=1000, ms=8,
                    markeredgecolor='w', label='Data')
        ymind, ymaxd = np.min(ydata), np.max(ydata)

    thisax.set_xlabel('x')
    thisax.set_ylabel('y')
    #thisax.set_xlim([np.min(xgrid), np.max(xgrid)])
    ymin, ymax = np.min(ygrid_pred['mean_mean']), np.max(ygrid_pred['mean_mean'])
    ymin, ymax = min(ymin, ymind), max(ymax, ymaxd)
    ymin, ymax = ymin - 0.15* (ymax-ymin), ymax + 0.15* (ymax-ymin)
    thisax.set_ylim([ymin, ymax])
    h, l = plt.gca().get_legend_handles_labels()
    thisax.legend(h[:3+idata], l[:3+idata], fontsize=14, ncol=1)
    #thisax.grid(False)
    plt.savefig('fit_1d_vars.png')


def plot_1dfit_shade(xgrid, ygrid_samples, xydata=None):
    """Plotting 1d fit with predictive variance shaded by quantiles.

    Args:
        xgrid (np.ndarray): An 1d array of x-grid
        ygrid_samples (np.ndarray): A 2d array of prediction samples.
        xydata (tuple, optional): If not None, this is the x- and y-data tuple and plots on top of the fits.
    """
    fig = plt.figure(figsize=(12, 9))
    thisax = plt.gca()
    plot_shade(xgrid, ygrid_samples, ax=thisax)

    ymind, ymaxd = -np.inf, np.inf
    idata = 0
    if xydata is not None:
        idata = 1
        xdata, ydata = xydata
        thisax.plot(xdata, ydata, 'ko', zorder=1000, ms=8,
                    markeredgecolor='w', label='Data')
        ymind, ymaxd = np.min(ydata), np.max(ydata)

    thisax.set_xlabel('x')
    thisax.set_ylabel('y')
    #thisax.set_xlim([np.min(xgrid), np.max(xgrid)])
    ymin, ymax = np.min(ygrid_samples), np.max(ygrid_samples)
    ymin, ymax = min(ymin, ymind), max(ymax, ymaxd)
    ymin, ymax = ymin - 0.15* (ymax-ymin), ymax + 0.15* (ymax-ymin)
    thisax.set_ylim([ymin, ymax])
    handles, labels = thisax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, fc='g'))
    labels.append('Posterior PDF')
    thisax.legend(handles[-1-idata:], labels[-1-idata:], fontsize=24, ncol=1)
    #thisax.grid(False)
    plt.savefig('fit_1d_shade.png')


def plot_ndfit_vars(ycheck, ycheck_pred):
    """Plots a diagonal plot of model (x-axis) and fit (y-axis)

    Args:
        ycheck (np.ndarray): An 1d array containing the model values.
        ycheck_pred (dict): Dictionary containing moments of predictions.
    """
    fig = plt.figure(figsize=(12, 9))
    thisax = plt.gca()
    variances = np.vstack((ycheck_pred['var_mean'], ycheck_pred['mean_var'])).T
    # np.savetxt('fvars.txt', variances)
    varlabels = ['Posterior uncertainty', 'Model error']
    varcolors = ['blue', 'lightblue']
    plot_vars(ycheck, ycheck_pred['mean_mean'], variances=variances, ysam=None,
                 stdfactor=1., varlabels=varlabels, varcolors=varcolors,
                 interp=None, connected=False, ax=thisax)
    thisax.set_xlabel('Model')
    thisax.set_ylabel('Fit')
    ymin1, ymax1 = np.min(ycheck_pred['mean_mean']), np.max(ycheck_pred['mean_mean'])
    ymin2, ymax2 = np.min(ycheck), np.max(ycheck)
    ymin, ymax = min(ymin1,ymin2), max(ymax1,ymax2)
    ymin, ymax = ymin - 0.05* (ymax-ymin), ymax + 0.05* (ymax-ymin)
    thisax.plot([ymin, ymax], [ymin ,ymax], 'k--')
    thisax.set_xlim([ymin, ymax])
    thisax.set_ylim([ymin, ymax])
    thisax.legend(fontsize=14, ncol=1)
    #thisax.grid(False)
    plt.savefig('fit_nd_vars.png')


