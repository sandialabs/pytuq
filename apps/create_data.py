#!/usr/bin/env python
"""This script generates training data for benchmark functions."""
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pytuq.func import toy, genz, chem, bench, poly, oper, func
from pytuq.utils.xutils import instantiate_classes_from_module


fcn_dict = {}
for submod in ['bench', 'bench1d', 'bench2d', 'benchNd', 'chem', 'genz', 'poly', 'toy']:
    this_objects = instantiate_classes_from_module(f"pytuq.func.{submod}")
    for j in this_objects:
        if j.name not in ['GenzBase', 'Poly']:
            fcn_dict[j.name] = j


usage_str = 'Script to create input-output data for benchmark functions.'
parser = argparse.ArgumentParser(description=usage_str)
#parser.add_argument('ind_show', type=int, nargs='*',
#                    help="indices of requested parameters (count from 0)")

parser.add_argument("-n", "--npts", dest="npts", type=int, default=100, help="Number of points")
parser.add_argument("-f", "--func", dest="func", type=str, default='Muller-Brown', help="Function name", choices=list(fcn_dict.keys()))
parser.add_argument("-x", "--xtrain", dest="xtrain_file", type=str, default=None, help="Optionally, provide x-sample file")
parser.add_argument("-s", "--sigma", dest="sig", type=float, default=0.0, help="Noise size")
parser.add_argument('-g', dest="grad", action='store_true',
                    help='Whether to compute gradients (default: False)')
parser.add_argument("-z", "--seed", dest="seed", type=int, default=None,
                    help="Seed for exact reproduction. If None, random seed is used.")

args = parser.parse_args()

fname = args.func
nsam = args.npts
grad = args.grad
sig = args.sig
xtrain_file = args.xtrain_file
seed = args.seed

if seed is not None:
    np.random.seed(seed)

try:
    fcn = fcn_dict[fname]
    print(f"{nsam} samples of function {fname} requested.")
except KeyError:
    print(f'Function {fname} is unknown. Please use -h to see the options. Exiting.')
    sys.exit()

if xtrain_file is None:
    xx = fcn.sample_uniform(nsam)
else:
    xx = np.loadtxt(xtrain_file)
    if len(xx.shape)==1:
        xx = xx.reshape(-1, 1)
    print(f"Input data file {xtrain_file} has {xx.shape[0]} samples, ignoring the -n flag.")

nsam, ndim = xx.shape
assert(ndim==fcn.dim)

yy = fcn(xx)
yy += sig * np.random.randn(*yy.shape)
print("Data noise sigma =", sig)
_, nout = yy.shape
assert(nout==fcn.outdim)


np.savetxt('xtrain.txt', xx)
np.savetxt('ytrain.txt', yy)
print("Input    data saved to xtrain.txt with shape ", xx.shape)
print("Output   data saved to ytrain.txt with shape ", yy.shape)
if grad:
    gg = fcn.grad(xx) #nsam, nout, ndim
    gg = gg.reshape(nsam, nout*ndim)
    np.savetxt('gtrain.txt', gg)
    print(f"Gradient data saved to gtrain.txt with shape ({nsam}, {nout}x{ndim})")


