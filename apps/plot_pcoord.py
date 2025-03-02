#!/usr/bin/env python

import os
import argparse
import numpy as np

from pytuq.utils.maps import Normalizer
from pytuq.utils.xutils import read_textlist
from pytuq.utils.plotting import parallel_coordinates, myrc

myrc()


usage_str = 'Script to plot parallel coordinates.'
parser = argparse.ArgumentParser(description=usage_str)
#parser.add_argument('ind_show', type=int, nargs='*',
#                    help="indices of requested parameters (count from 0)")
parser.add_argument("-x", "--xdata", dest="xdata", type=str, default='ptrain.txt',
                    help="Xdata file")
parser.add_argument("-y", "--ydata", dest="ydata", type=str, default=None,
                    help="Ydata file")
parser.add_argument("-o", "--outnames_file", dest="outnames_file", type=str, default='outnames.txt',
                    help="Output names file")
parser.add_argument("-p", "--pnames_file", dest="pnames_file", type=str, default='pnames.txt',
                    help="Param names file")
parser.add_argument("-e", "--every", dest="every", type=int, default=1,
                    help="Samples thinning")
parser.add_argument("-l", "--labels_file", dest="labels_file", type=str, default=None,
                    help="Label names file")
parser.add_argument("-c", "--ndcut", dest="ndcut", type=int, default=0,
                    help="Chunk size, if one wants to plot a parameter chunk at a time, useful for very high-d (if 0, plots all in one shot)")


args = parser.parse_args()


xdata = np.loadtxt(args.xdata)
scale = True

nsam, ndim = xdata.shape


pnames = read_textlist(args.pnames_file, ndim, names_prefix='par')

if args.ydata is not None:
    ydata = np.loadtxt(args.ydata)
    if len(ydata.shape)==1:
        ydata = ydata[:, np.newaxis]
    nsam_, nout = ydata.shape
    assert(nsam == nsam_)
    outnames = read_textlist(args.outnames_file, nout, names_prefix='out')

    xdata = np.hstack((xdata, ydata))
    #ndim += nout

if scale:
    sc = Normalizer(xdata)
    xdata = 2.*sc(xdata)-1.

if args.labels_file is not None:
    labels=read_textlist(args.labels_file, nsam)
else:
    labels=['']*nsam

ndcut= args.ndcut
if ndcut==0:
    ndcut=ndim
ndg=int((ndim-1)/ndcut)+1


ulabels = np.unique(labels)
labels=labels[::args.every]

for i in range(ndg):
    print("Plotting %d / %d " % (i+1,ndg))
    #names=range(1+i*ndcut,min(1+(i+1)*ndcut,ndim+1))

    values=xdata[::args.every,i*ndcut:min((i+1)*ndcut, ndim)].T

    pnames_this = pnames[i*ndcut:min((i+1)*ndcut, ndim)]
    if args.ydata is not None:
        values = np.vstack((values,xdata[::args.every,ndim:].T))
        pnames_this.extend(outnames)
    parallel_coordinates(pnames_this,
                         values, labels,'pcoord_'+str(i+1)+'.png')


    for lab in ulabels:
        labels_ = np.array(labels)[np.array(labels)==lab]
        parallel_coordinates(pnames_this,
                             values[:, np.array(labels)==lab],
                             list(labels_),
                             'pcoord_'+str(i+1)+'_lab'+lab+'.png')


    #labels_only=labels[labels==True]
    #values_only=values[:,labels==True]
    #io.parallel_coordinates(names, values_only, labels_only)
