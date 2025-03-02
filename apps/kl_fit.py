#!/usr/bin/env python


import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pytuq.linred.kle import KLE
from pytuq.linred.svd import SVD
from pytuq.utils.plotting import myrc, lighten_color, plot_dm, plot_xrv

myrc()


usage_str = 'Script to build PC surrogates of multioutput models.'
parser = argparse.ArgumentParser(description=usage_str)
parser.add_argument("-d", "--xcond", dest="xcond", type=str, default=None,
                    help="Xcond file")
parser.add_argument("-y", "--ydata", dest="ydata", type=str, default='ydata.txt',
                    help="Ydata file")
parser.add_argument("-e", "--neig", dest="neig", type=int, default=None,
                    help="Number of eigenvalues")
args = parser.parse_args()

####################################################################
xlabel= '' #r'log$_{10}$ Pressure'
ylabel = '' #'Amplitude'
plot_samfit = True
####################################################################

neig = args.neig

y = np.loadtxt(args.ydata)

if len(y.shape)==1:
    y = y[:, np.newaxis]
nsam, nout = y.shape


if args.xcond is None:
    xc = np.arange(nout)
else:
    xc = np.loadtxt(args.xcond)
    if len(xc.shape)>1:
        xc = np.arange(nout)


#lred = SVD()
lred = KLE()

lred.build(y.T)
if neig is None:
    neig = lred.get_neig(0.99)
ydata_red = lred.eval(neig=neig)


lred.plot_expvar()

if nout>=1000:
    nevery=100
else:
    nevery=1
for ix in range(0, nout, nevery):
    plot_dm([y[:,ix]], [ydata_red[ix,:]], errorbars=None, labels=['Training'], colors=None,
            axes_labels=['Model', 'Apprx'], figname=f'dm_{str(ix).zfill(3)}.png',
            legendpos='in', msize=7)
    plt.close(plt.gcf())

## Plot xi's
# plot_xrv(lred.xi[:, :10], prefix='xi')

# Plot to make sure data and data_red are close
if plot_samfit:
    if nsam>=1000:
        nevery=100
    else:
        nevery=1
    for isam in range(0, nsam, nevery):
        f = plt.figure(figsize=(12,4))
        plt.plot(xc, y[isam, :], 'bo-', ms=8, label='Model')
        plt.plot(xc, ydata_red.T[isam, :], 'go-', ms=8, label=f'KL (K={neig})')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(f'Sample #{isam+1}')
        plt.tight_layout()
        plt.savefig(f'fit_s{str(isam).zfill(3)}.png')
        #plt.savefig(f'fit_e{str(neig).zfill(3)}_s{str(isam).zfill(3)}.png')
        plt.close(f)
