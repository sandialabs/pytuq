#!/usr/bin/env python
"""App to build PC surrogates of multioutput models."""
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pytuq.utils.xutils import read_textlist, savepk
from pytuq.utils.plotting import myrc, lighten_color, plot_dm, plot_sens
#from pytuq.utils.maps import scale01ToDom

from pytuq.workflows.fits import pc_fit

myrc()


usage_str = 'App to build PC surrogates of multioutput models.'
parser = argparse.ArgumentParser(description=usage_str)
#parser.add_argument('ind_show', type=int, nargs='*',
#                    help="indices of requested parameters (count from 0)")
parser.add_argument("-x", "--xdata", dest="xdata", type=str, default='ptrain.txt',
                    help="Xdata file")
parser.add_argument("-y", "--ydata", dest="ydata", type=str, default='ytrain.txt',
                    help="Ydata file")
parser.add_argument("-d", "--xcond", dest="xcond", type=str, default=None,
                    help="Xcond file")
# parser.add_argument("-r", "--xrange", dest="xrange", type=str, default=None,
#                     help="Xrange file")
parser.add_argument("-q", "--outnames_file", dest="outnames_file", type=str, default='outnames.txt',
                    help="Output names file")
parser.add_argument("-p", "--pnames_file", dest="pnames_file", type=str, default='pnames.txt',
                    help="Param names file")
parser.add_argument("-t", "--trnfactor", dest="trnfactor", type=float, default=0.9,
                    help="Factor of data used for training")
parser.add_argument("-m", "--method", dest="method", type=str, default='bcs',
                    help="Fitting method", choices=['lsq', 'bcs', 'anl'])
parser.add_argument("-c", "--pctype", dest="pctype", type=str, default='LU',
                    help="PC type", choices=['LU', 'HG'])
parser.add_argument("-o", "--order", dest="order", type=int, default=1,
                    help="PC order.")

args = parser.parse_args()

####################################################################
xlabel='' #r'log$_{10}$ Pressure'
ylabel ='' # 'Amplitude'
plot_samfit = True
####################################################################

method = args.method
pctype = args.pctype
order = args.order
trnfactor = args.trnfactor


x = np.loadtxt(args.xdata)
y = np.loadtxt(args.ydata)


if len(x.shape)==1:
    x = x[:, np.newaxis]
if len(y.shape)==1:
    y = y[:, np.newaxis]

nsam, ndim = x.shape
nsam_, nout = y.shape

assert(nsam == nsam_)

if args.xcond is None:
    xc = np.arange(nout)
else:
    xc = np.loadtxt(args.xcond)
    if len(xc.shape)>1:
        xc = np.arange(nout)

outnames = read_textlist(args.outnames_file, nout, names_prefix='out')
pnames = read_textlist(args.pnames_file, ndim, names_prefix='par')

ntrn = int(trnfactor * nsam)

rperm = np.random.permutation(nsam)
indtrn = rperm[:ntrn]
indtst = rperm[ntrn:]

################################################################################
################################################################################


pcrv = pc_fit(x[indtrn], y[indtrn], order=order, pctype=pctype, method=method, eta=1.e-3)
ypred=pcrv.function(x)

if nout>=1000:
    nevery=100
else:
    nevery=1
for iout in range(0, nout, nevery):
    if len(indtst)==0:
        plot_dm([y[indtrn,iout]], [ypred[indtrn,iout]], errorbars=None, labels=['Training'], colors=None,
            axes_labels=['Model', 'Apprx'], figname=f'dm_{str(iout).zfill(3)}.png',
            legendpos='in', msize=7)
    else:
        plot_dm([y[indtrn,iout], y[indtst,iout]], [ypred[indtrn,iout], ypred[indtst,iout]], errorbars=None, labels=['Training', 'Testing'], colors=None,
            axes_labels=['Model', 'Apprx'], figname=f'dm_{str(iout).zfill(3)}.png',
            legendpos='in', msize=7)
    plt.close(plt.gcf())

if plot_samfit:
    if nsam>=1000:
        nevery=100
    else:
        nevery=1
    for isam in range(0, nsam, nevery):
        f = plt.figure(figsize=(12,4))
        plt.plot(xc, y[isam,:], 'bo-', ms=8, label='Model')
        plt.plot(xc, ypred[isam,:], 'go-', ms=8, label='PC apprx.')
        plt.title(f'Sample #{isam+1}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'fit_s{str(isam).zfill(3)}.png')
        plt.close(f)

mainsens = pcrv.computeSens()
pars = range(ndim)
cases = range(nout)

plot_sens(mainsens,pars,cases,vis="bar", ncol=5, par_labels=pnames, case_labels=[str(i) for i in cases], lbl_size=25, xticklabel_size=18, legend_size=18, figname='sens_pc.png')

# plot_sens(sensdata, pars, cases,
#               vis="bar", reverse=False, topsens=None,
#               par_labels=None, case_labels=None, colors=None,
#               xlbl='', title='', grid_show=True,
#               legend_show=2, legend_size=10, ncol=4,
#               lbl_size=22, yoffset=0.1,
#               xdatatick=None, xticklabel_size=None, xticklabel_rotation=0,
#               figname='sens.png')

savepk(pcrv, 'pcrv')
