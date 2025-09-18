#!/usr/bin/env python


import numpy as np

from pytuq.rv.pcrv import PCRV_iid, PCRV_mvn
from pytuq.utils.mindex import get_mi



# pdim = 5
# sdim = 3
# order = 4
# mi = get_mi(order, sdim)
# npc = mi.shape[0]
# cfs = [np.random.rand(npc) for j in range(pdim)]
# for j in range(pdim):
#     np.savetxt(f'cfs_{j}', cfs[j])
# mypcrv = PCRV(pdim, sdim, ["LU", "HG", "HG"], mi=mi, cfs=cfs)
# xx = mypcrv.sampleGerm(100)
# np.savetxt('xdata.dat', xx)
# #mypcrv.printInfo()
# yy = mypcrv.evalPC(xx)
# # mypcrv.sample(111)
# print(xx.shape, yy.shape)
# np.savetxt('ypy.dat', yy)
# print("=====================")
# print(mypcrv.coefs)
# cfs_flat = mypcrv.cfsFlatten()
# print("=====================")
# print(cfs_flat)
# mypcrv.cfsUnflatten(cfs_flat)
# print("=====================")
# print(mypcrv.coefs)


# pdim = 5
# #mypcrv = PCRV_iid(pdim, 'LU')
# mypcrv = PCRV_mvn(pdim)
# xx = mypcrv.sampleGerm(100)
# np.savetxt('xdata.dat', xx)
# mypcrv.printInfo()
# yy = mypcrv.evalPC(xx)
# # mypcrv.sample(111)
# print(xx.shape, yy.shape)
# np.savetxt('ypy.dat', yy)
# print(mypcrv.pind)
# print(mypcrv.computeMean())




pdim = 5
inpdf_type='pct'
pc_type='HG' #can be a list
rndind=[1, 2, 4]

if inpdf_type=='pci':
    orders = np.zeros(pdim, dtype=int)
    orders[rndind]=1
    mypcrv = PCRV_iid(pdim, pc_type, orders=orders)
elif inpdf_type=='pct':
    assert(pc_type=='HG')
    mypcrv = PCRV_mvn(pdim, rndind=rndind)
print(mypcrv.pind)
mypcrv.printInfo()
print(mypcrv.rndind, mypcrv.detind, mypcrv.sdim)
mypcrv.setRandomCfs()
print(mypcrv.coefs)
xx = mypcrv.sampleGerm(1000)
yy = mypcrv.evalPC(xx)


#setRandomCfs
cfs_flat=mypcrv.cfsFlatten()
print(cfs_flat)
mypcrv.cfsUnflatten(cfs_flat)
print(mypcrv.coefs)
