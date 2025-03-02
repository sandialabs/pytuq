#!/usr/bin/env python

import sys
import numpy as np

from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi
from pytuq.workflows.uprop import uprop_proj, uprop_regr

# Forward model examples
def cubic(xx):
    return xx**3

def sigmoid(xx):
    return 1./(1.+np.exp(-xx))

################################################################

# Dimensionality of the problem
dim = 3
model = cubic #sigmoid

# Setting up Input PC object
print("====== Input PC ======================================")
in_order = 1
in_mi = get_mi(in_order, dim)
in_cfs = np.random.rand(dim, in_mi.shape[0])

in_pc = PCRV(dim, dim, 'HG', mi=in_mi, cfs=in_cfs)
#in_pc.printInfo()
print(in_pc)
print(f"Means = {in_pc.computeMean()}")
print(f"Variances = {in_pc.computeVar()}")

# Setting up Output PC object
out_order = 5
out_mi = get_mi(out_order, dim)
out_pc =  PCRV(dim, dim, 'HG', mi=out_mi)

print("====== Output PC w projection =========================")
nqd = 7 # Number of quadrature points per dimension, usually need at least 2*out_order, but depends on model order
uprop_proj(in_pc, model, nqd, out_pc)
#out_pc.printInfo()
print(out_pc)
print(f"Means = {out_pc.computeMean()}")
print(f"Variances = {out_pc.computeVar()}")


print("====== Output PC w regression =========================")
nsam = 223 # Need at least as many sample as PC coefficients
uprop_regr(in_pc, model, nsam, out_pc)
#out_pc.printInfo()
print(out_pc)
print(f"Means = {out_pc.computeMean()}")
print(f"Variances = {out_pc.computeVar()}")

print("====== Moment estimation with plain Monte-Carlo ==========")
nsam = 111231 # Needs lots of samples for comparable accuracy
xsam = in_pc.sample(nsam)
ysam = model(xsam)

print(f"Means = {np.mean(ysam, axis=0)}")
print(f"Variances = {np.var(ysam, axis=0)}")
