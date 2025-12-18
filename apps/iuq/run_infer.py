#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np

try:
    import pprint
except ModuleNotFoundError:
    print("Please pip install pprint for more readable printing.")

import pytuq.gsa.gsa as gsa
import pytuq.minf.minf as minf
from pytuq.utils.xutils import savepk, loadpk, read_textlist



# Parse input arguments
usage_str = 'Workflow to perform Bayesian inference of a pre-constructed PC surrogate.'
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=usage_str)
parser.add_argument("-m", "--merr", dest="merr", action='store_true', default=False, help="Whether to infer with model error.")
args = parser.parse_args()


merr=args.merr

###############################################################
###############################################################
###############################################################

## Load surrogate results
results = loadpk('results') # really need to do LU
print("results.pk dictionary contents : ", list(results.keys()))

pcrv = results['pcrv']
germ_train, ptrain, ytrain, ytrain_pc, ytrain_pc_std, relerr_train = results['testing']
germ_test, ptest, ytest, ytest_pc, ytest_pc_std, relerr_test = results['testing']
allsens_main, allsens_total, allsens_joint = results['sens']


## Extra postprocessing of the surrogate to get the surrogate's empirical error metric
ntst = ytest.shape[0]
rmse_test = relerr_test*np.linalg.norm(ytest, axis=0)/np.sqrt(ntst)
surr_error_var1 = np.mean(ytest_pc_std**2, axis=0)
surr_error_var2 = rmse_test**2
surr_error_var = surr_error_var1+ surr_error_var2
np.savetxt('surr_error_var.txt', surr_error_var)


print(f"Number of inputs  : {pcrv.function.dim}")
print(f"Number of outputs : {pcrv.function.outdim}")

###############################################################
###############################################################
###############################################################

## Load data
ydata = np.loadtxt('ydata.txt')
if len(ydata.shape)==1:
    ydata = ydata.reshape(-1, 1)
ydatavar = np.loadtxt('ydatavar.txt')
nx = ydata.shape[0]
xdata = np.arange(nx)

pdim = pcrv.function.dim
nx_ = pcrv.function.outdim
nx__ = ydata.shape[0]
assert(nx_==nx)
assert(nx__==nx)

###############################################################
###############################################################
###############################################################


## Configure Inference

# Only calibrate parameters with high enough sensitivity
ind_infer = np.arange(pdim)[np.mean(allsens_total, axis=0) > 0.01]
nfix = pdim - len(ind_infer)
fixindnom_val = [0.0]*nfix
ind_fix = np.setdiff1d(np.arange(pdim), ind_infer)
fixindnom = list(zip(ind_fix, fixindnom_val))

if merr:
    rndind = range(pdim)
    lik_type = 'gausmarg'
    outord = 2
else:
    rndind = []
    lik_type = 'classical'
    outord = 0

# Indices of data to be used for calibration
xind_calib = range(nx)  #[1, 3] #range(nx)

calib_params={'param_ini': None, 'cov_ini': None,
't0': 100, 'tadapt' : 100, 'gamma' : 0.1, 'nmcmc' : 1000}

## Model parameter inference configuration
model_infer_config = {
    "inpdf_type" : 'pci', "pc_type" : 'LU', "outord" : outord,
    "rndind" : rndind, "fixindnom" : fixindnom, "ind_calib" : xind_calib,
    "calib_type" : 'amcmc', "calib_params" : calib_params,
    "md_transform" : None, "zflag" : True, "datamode" : None,
    "lik_type" : lik_type, "lik_params" : {},
    "pr_type" : 'uniform', "pr_params" : {},
    "dv_type" : 'var_fixed', "dv_params" : [surr_error_var+ydatavar]
}


## Beautified printing
try:
    pprint.pprint(model_infer_config, indent=1, compact=True)
except NameError:
    print(model_infer_config)


###############################################################
###############################################################
###############################################################

## Run Inference

calib_model = lambda p, x: pcrv.function(p)
calib_model_params = None

calib_results = minf.model_infer(ydata[xind_calib],
                                 calib_model, calib_model_params, pdim,
                                 **model_infer_config)


## Save results

np.savetxt('chain.txt', calib_results['chain'])
np.savetxt('mapparams.txt', calib_results['mapparams'].reshape(1, -1))
savepk(calib_results, 'calib_results')
print("Calibration results saved in dictionary results.pk")

print("Calibration results dictionary contents : ", list(calib_results.keys()))
