#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np

from pytuq.utils.mindex import get_mi
from pytuq.utils.xutils import safe_cholesky
from pytuq.workflows.fits import pc_ros

################################################################################
################################################################################
################################################################################

usage_str = 'Input PC generation given mvn, samples or marginal PC.'
parser = argparse.ArgumentParser(description=usage_str)
parser.add_argument("-f", "--fmt", dest="input_format", type=str, default='sam', help="Input format", choices=['marg', 'sam', 'mvn'])
parser.add_argument("-i", "--inp", dest="filename", type=str, default='xsam.txt', help="Input filename: marginal coefficients (if format is marg), samples (if format is sam), mean (if format is mvn).")
parser.add_argument("-c", "--cov", dest="cov_filename", type=str, default='cov.txt', help="Covariance filename (relevant if format is mvn).")
parser.add_argument("-p", "--pco", dest="pcorder", type=int, default=1, help="PC order (relevant if format is marg or sam).")
parser.add_argument("-t", "--pct", dest="pctype", type=str, default='HG', help="PC type (relevant if format is sam).")
args = parser.parse_args()

################################################################################
################################################################################
################################################################################

input_format=args.input_format
filename=args.filename

if input_format == "marg" or input_format == "sam":
	pcorder=args.pcorder
if input_format == "mvn":
	cov_filename=args.cov_filename
if input_format == "sam":
	pctype = args.pctype


################################################################################
################################################################################
################################################################################

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


	assert(pcorder >= maxord)
	mindex_totalorder=get_mi(pcorder, dim)

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
	sams = np.loadtxt(filename)

	nsam, dim = sams.shape
	pcrv = pc_ros(sams, pctype=pctype, order=pcorder, nreg=nsam, bwfactor=1.0)
	np.savetxt('pcf.txt', np.array(pcrv.coefs).T)

elif input_format=="mvn":
	mean = np.loadtxt(filename)
	cov = np.loadtxt(cov_filename)

	dim = mean.shape[0]


	lower = safe_cholesky(cov)

	param_pcf = np.zeros((dim + 1, dim))
	param_pcf[0, :] = mean
	param_pcf[1:, :] = lower.T

	np.savetxt('pcf.txt', param_pcf)

else:
	print("pc_prep.py : Input format not recognized. Must be marg or sam or mvn.")
