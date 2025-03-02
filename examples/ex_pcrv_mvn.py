#!/usr/bin/env python


import numpy as np

from pytuq.rv.pcrv import PCRV_iid, PCRV_mvn
from pytuq.utils.mindex import get_mi


covMatSize=14 #dimension of L and C
pcfs = np.random.rand(covMatSize, covMatSize+1) #np.loadtxt('pcf.txt').T

#pcrv_mvn = PCRV_mvn(pdim=covMatSize,cfs=pcfs)

# pcrv_mvn = PCRV_mvn(pdim=covMatSize)
# pcrv_mvn.setCfs(cfs=[pcfs[j,:j+2] for j in range(covMatSize)])

pcrv_mvn = PCRV_mvn(pdim=covMatSize, mean=pcfs[:,0], cov=np.dot(pcfs[:,1:], pcfs[:,1:].T))
#pcrv_mvn = PCRV_mvn(pdim=covMatSize)
#pcrv_mvn.setCfs(cfs=[pcfs[j,:j+2] for j in range(covMatSize)])

output_samples = pcrv_mvn.sample(20)

pcrv_mvn.printInfo()
# pdim = 14

# #cfs = np.random.rand(pdim, pdim+1)
# mypcrv = PCRV_mvn(pdim)

#mypcrv.printInfo()

# mypcrv.setRandomCfs()
# print(mypcrv.coefs)
# xx = mypcrv.sampleGerm(20)
# yy = mypcrv.evalPC(xx)
