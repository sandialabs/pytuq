#!/usr/bin/env python

import sys
import argparse
import numpy as np

import scipy.stats as ss
import matplotlib.pyplot as plt

from pytuq.utils.plotting import myrc, plot_cov, plot_cov_tri

myrc()

usage_str = 'Script to plot multivariate normal 2d marginal covariances.'
parser = argparse.ArgumentParser(description=usage_str)
parser.add_argument('ind_show', type=int, nargs='*',
                    help="indices of requested parameters (count from 0)")
parser.add_argument("-m", "--mean", dest="mean", type=str, default='mean.txt',
                    help="Mean file")
parser.add_argument("-c", "--cov", dest="cov", type=str, default='cov.txt',
                    help="Covariance file")
args = parser.parse_args()

mean = np.loadtxt(args.mean)
cov = np.loadtxt(args.cov)
#cov = np.eye(3)
#print(np.linalg.eigvals(cov))

dim = mean.shape[0]

if len(args.ind_show)==0:
    ind_show = range(dim)
else:
    ind_show = args.ind_show

dim_show = len(ind_show)


for ii in range(dim_show):
    for jj in range(ii+1,dim_show):
        i, j = ind_show[ii], ind_show[jj]

        mm = np.array([mean[i], mean[j]])
        cc = np.array([[cov[i,i], cov[i,j]],[cov[j,i], cov[j,j]]])
        plot_cov(mm, cc, f=3., pnames=[f'p{i}', f'p{j}'], ngr=100, savefig=True)
        plt.clf()

plot_cov_tri(mean[ind_show], cov[np.ix_(ind_show, ind_show)], names=[f'p{i}' for i in ind_show])

