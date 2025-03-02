#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from xuq.plotting import plot_yx, myrc
from xuq.myutils import read_textlist

myrc()

usage_str = 'Script to plot data pair of dimensions at a time.'
parser = argparse.ArgumentParser(description=usage_str)
#parser.add_argument('ind_show', type=int, nargs='*',
#                    help="indices of requested parameters (count from 0)")
parser.add_argument("-x", "--xdata", dest="xdata", type=str, default='qtrain.txt',
                    help="Xdata file")
parser.add_argument("-p", "--pnames_file", dest="pnames_file", type=str, default='pnames.txt',
                    help="Param names file")
parser.add_argument("-e", "--every", dest="every", type=int, default=1,
                    help="Samples thinning")
parser.add_argument("-l", "--labels_file", dest="labels_file", type=str, default=None,
                    help="Label names file")


args = parser.parse_args()


xdata = np.loadtxt(args.xdata)
if len(xdata.shape)==1:
    print("Nothing to plot: the dataset is one-dimensional")

nsam, ndim = xdata.shape

pnames = read_textlist(args.pnames_file, ndim, names_prefix='par')

if args.labels_file is not None:
    labels=read_textlist(args.labels_file, nsam)
else:
    labels=['']*nsam

labels=labels[::args.every]
xdata = xdata[::args.every, :]
labels = np.array(labels)
ulabels = np.unique(labels)


for idim in range(ndim):
    for jdim in range(idim+1, ndim):
        xname = pnames[idim]
        yname = pnames[jdim]
        print(xname, yname)

        plt.figure(figsize=(10,9))



        for lab in ulabels:
            plt.plot(xdata[labels==lab, idim], xdata[labels==lab, jdim], 'o', label=lab, alpha=0.5)
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.legend()
        plt.savefig(f'xx_{xname}_{yname}.png')
        plt.clf()

# rows = 1
# cols = len(ulabels)
# for idim in range(ndim):
#     for jdim in range(idim+1, ndim):
#         xname = pnames[idim]
#         yname = pnames[jdim]
#         fig, axes = plt.subplots(rows, cols, figsize=(10*cols,10*rows),
#                              gridspec_kw={'hspace': 0.0, 'wspace': 0.3})
#         for i, lab in enumerate(ulabels):
#             axes[i].plot(xdata[labels==lab, idim], xdata[labels==lab, jdim], 'go')
#             axes[i].set_title(f'Group {lab}')
#             axes[i].set_xlabel(xname)
#             axes[i].set_ylabel(yname)
#         plt.savefig(f'xx_{xname}_{yname}_grouped.png')
#         plt.clf()
