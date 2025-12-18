#!/usr/bin/env python
"""Example demonstrating uncertainty propagation through a model with PC inputs.

This script shows how to propagate polynomial chaos input uncertainties through
a nonlinear model using projection or regression methods.
"""

import numpy as np

from pytuq.rv.pcrv import PCRV, PCRV_iid
from pytuq.utils.mindex import get_mi

#method='projection'
method='regression'

def model(p):
    return np.sum(p, axis=1)**2


outord = 2
pdim = 5
inpc = PCRV_iid(pdim, 'HG')
print("Coefficient's (parameter,index) pairs: ")
print(inpc.pind)


cfs_all = []
for i in range(pdim):
    cfs = np.zeros(inpc.coefs[i].shape[0])
    cfs[-1] = 1.0
    cfs_all.append(cfs)
inpc.setCfs(cfs_all)
print("Input PC (multiindex,coefficient) pairs: ")
inpc.printInfo()

mindex = get_mi(outord, inpc.sdim)
outpc = PCRV(1, inpc.sdim, 'HG', mi=mindex, cfs=None)
outnorms = outpc.evalBasesNormsSq(0)


if method=="projection":
    # os.system('generate_quad -d 5 -g HG -x full -p 5')
    # xx = np.loadtxt('qdpts.dat')
    # ww = np.loadtxt('wghts.dat')
    xx, ww = outpc.quadGerm([4] * inpc.sdim)
elif method=="regression":
    xx = inpc.sampleGerm(100)

Amat = outpc.evalBases(xx, 0)
pp = inpc.evalPC(xx)


yy = model(pp)



if method=="projection":
    cfs = np.dot(Amat.T, ww * yy) / outnorms
elif method=="regression":
    invptp = np.linalg.inv(np.dot(Amat.T, Amat))
    cfs = np.dot(invptp, np.dot(Amat.T, yy))

outpc.setCfs(cfs=[cfs])
print("Output PC (multiindex,coefficient) pairs: ")
outpc.printInfo()

# print(outpc.computeMean())
# print(outpc.coefs)
# print(outpc.evalBasesNormsSq(0))

