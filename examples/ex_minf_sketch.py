#!/usr/bin/env python
"""An example parameter inference sketch for 1-dimensional data."""

try:
    import pprint
except ModuleNotFoundError:
    print("Please pip install pprint for more readable printing.")

import numpy as np

import pytuq.func.func as func
import pytuq.minf.minf as minf
import pytuq.utils.plotting as pl
import pytuq.utils.stats as st

pl.myrc()

def const_model(pp, xx):
    yy = np.empty((pp.shape[0], xx.shape[0]))
    for i in range(len(pp)):
        yy[i, :] = pp[i, 0]+0.0
    return yy

def exp_model_single(par, xx):
    return par[1] * np.exp(par[0] * xx) - 2

def exp_model(pp, xx):
    yy = np.empty((pp.shape[0], xx.shape[0]))
    for i in range(len(pp)):
        yy[i, :] = exp_model_single(pp[i, :], xx).T
    return yy

def linear_model_single(par, xx):
    return par[0] + par[1] * xx

def linear_model(pp, xx):
    yy = np.empty((pp.shape[0], xx.shape[0]))
    for i in range(len(pp)):
        yy[i, :] = linear_model_single(pp[i, :], xx).T
    return yy

def tanh_model(xx):
    return np.tanh(3 * (xx - 0.3))

def lin_model(xx):
    return 2.*xx-3.

########################################################################
########################################################################
########################################################################

## Set up true model
true_model = tanh_model #lin_model #tanh_model
domain = None #np.array([[0.6, 1.0], [1.4, 1.7]]) #None

## Set up a grid of input for plotting
ngrid = 33
xgrid = np.linspace(-1, 1, ngrid)

## X-data locations where data observations are taken
npt = 15
xdata = 0.7 * np.linspace(-1., 1., npt).reshape(-1, 1)



## Generate true model output on a grid, and noisy data from the true model at x-data inputs
ygrid = true_model(xgrid)
datanoise_sig = 0.1
ydata = true_model(xdata) + datanoise_sig * np.random.randn(npt, 1)

## Plot data and true model only
minf.plot_1d_data(xdata, ydata, xygrid_true=(xgrid, ygrid))


## Model to be calibrated
calib_model = exp_model #const_model #linear_model #exp_model #const_model  #linear_model #exp_model
calib_model_params = xdata
calib_model_pdim = 2
fixindnom = [] #[[1, 0.5], [0, -0.5]] # ok if not ordered
md_transform=None

## If there is a parameter domain, plot samples of the true model from the domain prior
if domain is not None:
    prior_psamples = np.random.rand(100, 2)
    prior_psamples_phys = prior_psamples*(domain[:,1]-domain[:,0])+domain[:,0]
    prior_fsamples = calib_model(prior_psamples_phys, xgrid)
    minf.plot_1d_samples(xdata, ydata, xygrid_samples=(xgrid, prior_fsamples.T))




## Likelihood options
# lik_type, lik_params, rndind= 'dummy', {}, []
# lik_type, lik_params, rndind= 'classical', {}, []
# lik_type, lik_params, rndind= 'abc', {'abceps': 0.1, 'abcalpha':1.0}, [0]
lik_type, lik_params, rndind= 'gausmarg', {}, [0, 1]

## Prior options
pr_type, pr_params = 'uniform', {'domain': domain}
# pr_type, pr_params = 'normal', {'mean': 2.*np.ones(4), 'var': None}

## Data-variance options
dv_type, dv_params = 'var_fixed', [datanoise_sig**2]
#dv_type, dv_params = 'std_infer', []

## Calibration method options
calib_params={'param_ini': None, 'cov_ini': None,
't0': 100, 'tadapt' : 100, 'gamma' : 0.1, 'nmcmc' : 10000}

## Model parameter inference configuration
model_infer_config = {
    "inpdf_type" : 'pci', "pc_type" : 'LU', "outord" : 1,
    "rndind" : rndind, "fixindnom" : fixindnom, "ind_calib" : None,
    "calib_type" : 'amcmc', "calib_params" : calib_params,
    "md_transform" : md_transform, "zflag" : True, "datamode" : None,
    "lik_type" : lik_type, "lik_params" : lik_params,
    "pr_type" : pr_type, "pr_params" : pr_params,
    "dv_type" : dv_type, "dv_params" : dv_params
}

## Beautified printing
try:
    pprint.pprint(model_infer_config, indent=1, compact=True)
except NameError:
    print(model_infer_config)

## Run the inference
calib_results = minf.model_infer(ydata,
                                 calib_model, calib_model_params, calib_model_pdim,
                                 **model_infer_config)


elpost = lambda xx: np.array([calib_results['post'].evalLogPost(x) for x in xx])
lpost = func.ModelWrapperFcn(elpost, calib_results['chain'].shape[1])
post_domain = st.get_domain(calib_results['chain'])
lpost.setDimDom(domain=post_domain)
lpost.plot_2d(ngr=55)

## Save chain for debugging
np.savetxt('chain.txt', calib_results['chain'])
np.savetxt('logpost.txt', calib_results['logpost'])
np.savetxt('alphas.txt', calib_results['alphas'])


## Postprocess model at grid points for plotting
checkmode = [calib_results['post'].model, xgrid, range(len(xgrid)), md_transform]
ycheck_pred, samples =  minf.model_infer_postp(calib_results,
                                                          checkmode=checkmode,
                                                          nburn=len(calib_results['chain'])//10,
                                                          nevery=10, nxi=1)


print("MAP:", calib_results['mapparams'])

psamples, fsamples = samples['pmcmc'], samples['fmcmc']

## More plots to visualize fits
minf.plot_ndfit_vars(ygrid, ycheck_pred)
minf.plot_1dfit(xgrid, ycheck_pred, ygrid_true=ygrid, xydata=(xdata,ydata))
minf.plot_1dfit_vars(xgrid, ycheck_pred, xydata=(xdata,ydata))
fsamples_2d = np.reshape(fsamples, (-1, ngrid), order='F') #nsam, ngrid
minf.plot_1dfit_shade(xgrid, fsamples_2d.T, xydata=(xdata,ydata))
print(psamples.shape)
psamples_2d = np.reshape(psamples, (-1, calib_model_pdim), order='F') #nsam, npar


## Visualize parameter posteriors in a few ways
pl.plot_pdfs(ind_show=None, samples_=psamples_2d,
          plot_type='ind', names_=None,
          nominal_=None, prange_=domain,lsize=13)

pl.plot_pdfs(ind_show=None, samples_=psamples_2d,
          plot_type='inds', names_=None,
          nominal_=None, prange_=domain,lsize=13)

# ## Example of evaluating another model with the calibrated parameters
# def new_model(pp, xx):
#     yy = np.empty((pp.shape[0], xx.shape[0]))
#     for i in range(len(pp)):
#         yy[i, :] = np.sin(pp[i, 0]*xx) + pp[i, 1]*xx
#     return yy

# ## Example of evaluating another model
# ngrid = 100
# xgrid = np.linspace(0, 10, ngrid)
# checkmode = [new_model, xgrid, range(len(xgrid)), None]
# ycheck_pred, psamples, fsamples =  minf.model_infer_postp(calib_results,
#                                                           checkmode=checkmode,
#                       nburn=len(calib_results['chain'])//10, nevery=10, nxi=1)


# minf.plot_1dfit(xgrid, ycheck_pred)
# minf.plot_1dfit_vars(xgrid, ycheck_pred)
# fsamples_2d = np.reshape(fsamples, (-1, ngrid), order='F') #nsam, ngrid
# minf.plot_1dfit_shade(xgrid, fsamples_2d.T)

