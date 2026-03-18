#!/usr/bin/env python
"""Create synthetic PC-based observational data for inference testing.

This script loads a previously built PC surrogate (from ``results.pk``),
evaluates it at a true parameter vector, adds Gaussian noise, and writes
out synthetic data and variance files for use by the inference workflow.

Outputs:
    ``p_true.txt``    : True parameter vector used for data generation.
    ``ydata.txt``     : Synthetic observational data of shape ``(outdim, neach)``.
    ``ydatavar.txt``   : Data-noise variance array of shape ``(outdim,)``.

Example::

    python create_data_truepc.py -s 0.5 -e 3 -c p_true_input.txt

Command-line arguments:
    -s, --sig     Noise standard deviation (default: 0.5).
    -e, --each    Number of replicate samples per output location (default: 1).
    -c, --cfile   File of true coefficients; if omitted, random values are drawn.
"""

import argparse
import numpy as np

from pytuq.utils.xutils import loadpk


# Parse input arguments
usage_str = 'Workflow to create synthetic PC-based data according to the results of the uqpc workflow.'
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=usage_str)
parser.add_argument("-s", "--sig", dest="datasig", type=float, default=0.5,
                    help="Noise standard deviation")
parser.add_argument("-e", "--each", dest="neach", type=int, default=1,
                    help="Number of samples per x-location")
parser.add_argument("-c", "--cfile", dest="cf_true_file", type=str, default=None, help="Coeff file name. Defaults to None, in which case a random set of coefficients is selected.")

args = parser.parse_args()


datasig=args.datasig
neach=args.neach
cf_true_file=args.cf_true_file

results = loadpk('results')
pcrv = results['pcrv']

print(f"Number of inputs  : {pcrv.function.dim}")
print(f"Number of outputs : {pcrv.function.outdim}")

p_true = np.loadtxt(cf_true_file) if cf_true_file is not None else np.random.rand(pcrv.function.dim)
print("Using true coefficient values : ", p_true)
np.savetxt('p_true.txt', p_true)

ydata = pcrv.function(p_true.reshape(1,-1))+datasig*np.random.randn(neach,pcrv.function.outdim)
ydata = ydata.T # Transpose to shape (outdim, neach)

ydatavar = (datasig**2)*np.ones((pcrv.function.outdim))

np.savetxt('ydata.txt', ydata)
np.savetxt('ydatavar.txt', ydatavar)


