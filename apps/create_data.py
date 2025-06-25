#!/usr/bin/env python
"""This script generates training data for benchmark functions."""
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pytuq.func import toy, genz, chem, benchmark, poly, oper, func




usage_str = 'Script to create input-output data for benchmark functions.'
parser = argparse.ArgumentParser(description=usage_str)
#parser.add_argument('ind_show', type=int, nargs='*',
#                    help="indices of requested parameters (count from 0)")

parser.add_argument("-n", "--npts", dest="npts", type=int, default=100, help="Number of points")
parser.add_argument("-f", "--func", dest="func", type=str, default='lj', help="Function name", choices=['lj', 'mb'])


args = parser.parse_args()

fname = args.func
nsam = args.npts

if fname == 'lj':
    fcn = chem.LennardJones()
elif fname == 'mb':
    fcn = chem.MullerBrown()
else:
    print(f'Function {fcn} is unknown. Please use -h to see the options. Exiting.')
    sys.exit()

xx = fcn.sample_uniform(nsam)

yy = fcn(xx)
gg = fcn.grad(xx) #nsam, nout, ndim
nsam_, nout, ndim = gg.shape
gg = gg.reshape(nsam, nout*ndim)

np.savetxt('xtrain.txt', xx)
np.savetxt('ytrain.txt', yy)
np.savetxt('gtrain.txt', gg)
