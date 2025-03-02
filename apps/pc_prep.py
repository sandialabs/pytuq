#!/usr/bin/env python

################################################################################
# Input PC generation given mvn, samples or marginal pc
################################################################################



# Input
# Usage  : pc_prep.py <format> <filename> <input_pcorder>
# e.g.   : pc_prep.py marg marg_pc.txt 3
#        : pc_prep.py sam inp_sam.txt 3
#        : pc_prep.py mvn mean.txt cov.txt
# Output : pcf.txt


import os
import sys
import numpy as np

from pytuq.utils.mindex import get_mi
from pytuq.utils.xutils import safe_cholesky

input_format=sys.argv[1]
filename=sys.argv[2]

if input_format == "marg" or input_format == "sam":
	input_pcorder=int(sys.argv[3])
elif input_format == "mvn":
	filename2=sys.argv[3]


if input_format=="marg":

	with open(filename) as f:
	    margpc = f.readlines()
	dim=len(margpc)



	margpc_all=[]
	maxord=0
	sumord=0
	for i in range(dim):
		margpc_cur=np.array(margpc[i].split(),dtype=float)
		order=margpc_cur.shape[0]-1
		if maxord<order:
			maxord=order
		sumord+=order
		margpc_all.append(margpc_cur)


	assert(input_pcorder >= maxord)
	mindex_totalorder=get_mi(input_pcorder, dim)

	mindex=np.zeros((1,dim),dtype=int)
	cfs=np.zeros((1,dim))
	for i in range(dim):
		cfs[0,i]=margpc_all[i][0]

	for i in range(dim):

		order_this=margpc_all[i].shape[0]-1
		mindex_this=np.zeros((order_this,dim),dtype=int)
		mindex_this[:,i]=np.arange(1,order_this+1)
		mindex=np.vstack((mindex,mindex_this))

		cfs_this=np.zeros((order_this,dim))
		cfs_this[:,i]=margpc_all[i][1:]
		cfs=np.vstack((cfs,cfs_this))

	k=0
	cfs_totalorder=np.zeros(mindex_totalorder.shape)
	for mi in mindex_totalorder:
		if any(np.equal(mi,mindex).all(1)):
			ind=np.equal(mi,mindex).all(1).tolist().index(True)
			cfs_totalorder[k,:]=cfs[ind,:]
		k+=1

	np.savetxt('pcf.txt',cfs_totalorder)


elif input_format=="sam":
	sam = np.loadtxt(filename)
	ros = Rosenblatt(sam)
	print("Not implemented yet")
	sys.exit()
	# TODO: make ex_ros_pc.py a function and use it here

	# cmd=uqtkbin+'pce_quad -o '+str(input_pcorder)+' -w HG -f '+filename+ ' > pcq.log; mv PCcoeff.dat pcf.txt'
	# os.system(cmd)

elif input_format=="mvn":
	mean = np.loadtxt(filename)
	cov = np.loadtxt(filename2)

	dim = mean.shape[0]


	lower = safe_cholesky(cov)

	param_pcf = np.zeros((dim + 1, dim))
	param_pcf[0, :] = mean
	param_pcf[1:, :] = lower.T

	np.savetxt('pcf.txt', param_pcf)

else:
	print("pc_prep.py : Input format not recognized. Must be marg or sam.")
