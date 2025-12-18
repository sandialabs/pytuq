#!/usr/bin/env python
"""Example demonstrating PC-based global sensitivity analysis.

This script computes Sobol sensitivity indices using polynomial chaos expansions,
optionally building a PC surrogate first or directly using the Ishigami function.
"""


try:
    import pprint
except ModuleNotFoundError:
    print("Please pip install pprint for more readable printing.")

from pytuq.lreg.lreg import lsq
from pytuq.func.bench import Ishigami
from pytuq.gsa.gsa import PCSobol
from pytuq.utils.plotting import plot_sens, plot_jsens, myrc

myrc()

LUsurr = False # whether to preconstruct a surrogate

# Setup the function of interest
myfunc = Ishigami()

# Number of samples
nsam = 100000

if LUsurr:
    # this is an overkill, we could directly use pcrv and not PCSobol
    sMethod_LU = PCSobol(myfunc.domain, pctype='LU', order=7)
    # Sample the input
    xsam = sMethod_LU.sample(nsam)
    # Evaluate the function
    ysam = myfunc(xsam)

    Amat = sMethod_LU.pcrv.evalBases(sMethod_LU.germ_sam, 0)
    lreg = lsq()
    lreg.fita(Amat, ysam[:,0])
    sMethod_LU.pcrv.setCfs([lreg.cf])
    sMethod_LU.pcrv.setFunction()
    myfunc = sMethod_LU.pcrv.function

sMethod = PCSobol(myfunc.domain, pctype='HG', order=3)

# Sample the input
xsam = sMethod.sample(nsam)
# Evaluate the function
ysam = myfunc(xsam)
# Compute the sensitivities
sens = sMethod.compute(ysam)

# Print the sensitivities
try:
    pprint.pprint(sens)
except NameError:
    print(sens)

# Plot main sensitivities
plot_sens(sens['main'].reshape(1,-1),range(myfunc.dim),range(1))
# Plot joint sensitivities
plot_jsens(sens['main'], sens['jointt'])
