#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pytuq.utils.plotting import plot_yx
from pytuq.utils.xutils import read_textlist


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
parser.add_argument("-c", "--cols", dest="cols", type=int, default=4,
                    help="Num of columns")
parser.add_argument("-r", "--rows", dest="rows", type=int, default=6,
                    help="Num of rows")


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



for iout in range(nout):
    outname = outnames[iout]
    print(outname)
    plot_yx(xdata[::args.every], ydata[::args.every, iout], rowcols=(args.rows, args.cols), ylabel=outname, xlabels=pnames, log=False, filename='yx_'+outname+'.png', ypad=1.2, gridshow=False, ms=6, labelsize=16)
    plot_yx(xdata[::args.every], ydata[::args.every, iout], rowcols=(args.rows, args.cols), ylabel=outname, xlabels=pnames, log=True, filename='yx_'+outname+'_log.png', ypad=1.2, gridshow=False, ms=6, labelsize=16)
    plt.gcf().clear()

