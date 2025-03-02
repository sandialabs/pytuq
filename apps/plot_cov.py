#!/usr/bin/env python

import sys
import argparse
import numpy as np

import scipy.stats as ss
import matplotlib.pyplot as plt

from pytuq.utils.plotting import myrc

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


f=3.

for ii in range(dim_show):
    for jj in range(ii+1,dim_show):
        i, j = ind_show[ii], ind_show[jj]
        x = np.linspace(mean[i]-f*np.sqrt(cov[i,i]), mean[i]+f*np.sqrt(cov[i,i]), 100)
        y = np.linspace(mean[j]-f*np.sqrt(cov[j,j]), mean[j]+f*np.sqrt(cov[j,j]), 100)
        X, Y = np.meshgrid(x, y)

        try:
            rv = ss.multivariate_normal([mean[i], mean[j]], [[cov[i,i], cov[i,j]],[cov[j,i], cov[j,j]]], allow_singular=True)
            XY = np.dstack((X, Y))

            Z = rv.pdf(XY)
            plt.contour(X,Y,Z)
            plt.xlabel('p'+str(i+1))
            plt.ylabel('p'+str(j+1))
            plt.savefig('cov_'+str(i)+'_'+str(j)+'.png')
            plt.clf()

        except ValueError:
            print(f"Covariance for pair ({i},{j}) is not positive-semidefinite.")
