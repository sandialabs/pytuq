#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pytuq.utils.plotting import plot_tri, myrc
from pytuq.utils.xutils import read_textlist

myrc()

usage_str = 'Script to plot outputs wrt one input at a time.'
parser = argparse.ArgumentParser(description=usage_str)
#parser.add_argument('ind_show', type=int, nargs='*',
#                    help="indices of requested parameters (count from 0)")
parser.add_argument("-x", "--xdata", dest="xdata", type=str, default='qtrain.txt',
                    help="Xdata file")
parser.add_argument("-y", "--ydata", dest="ydata", type=str, default='ytrain.txt',
                    help="Ydata file")
parser.add_argument("-o", "--outnames_file", dest="outnames_file", type=str, default='outnames.txt',
                    help="Output names file")
parser.add_argument("-p", "--pnames_file", dest="pnames_file", type=str, default='pnames.txt',
                    help="Param names file")
parser.add_argument("-e", "--every", dest="every", type=int, default=1,
                    help="Samples thinning")



args = parser.parse_args()


xdata = np.loadtxt(args.xdata)
ydata = np.loadtxt(args.ydata)
if len(ydata.shape)==1:
    ydata = ydata.reshape(-1,1)

nsam, ndim = xdata.shape
nsam_, nout = ydata.shape
assert(nsam == nsam_)

outnames = read_textlist(args.outnames_file, nout, names_prefix='out')
pnames = read_textlist(args.pnames_file, ndim, names_prefix='par')

#def plot_tri(xi, names=None, msize=3, axarr=None, clr='b', zorder=None, figname=None):


for iout in range(nout):
    outname = outnames[iout]
    print(f"Plotting pairwise samples for output {outname}")
    plot_tri(xdata, yy=ydata[:, iout], names=pnames, msize=3, figname=f'yxx_{iout}.png')
    plt.gcf().clear()

