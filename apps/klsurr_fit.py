#!/usr/bin/env python
"""App to build KL-pbased reduced-dimensional surrogates of multioutput models."""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pytuq.linred.klsurr import KLSurr
from pytuq.rv.pcrv import PCRV

from pytuq.utils.xutils import read_textlist, savepk
from pytuq.utils.plotting import myrc, plot_dm, plot_sens
from pytuq.utils.stats import get_domain

myrc()
#plt.rcParams.update({'figure.max_open_warning': 2000})


usage_str = 'App to build KL-based reduced-dimensional surrogates of multioutput models.'
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
parser.add_argument("-p", "--pnames_file", dest="pnames_file", type=str, default='pnames.txt',                    help="Param names file")
parser.add_argument("-t", "--trnfactor", dest="trnfactor", type=float, default=0.9,
                    help="Factor of data used for training")
parser.add_argument("-s", "--surr", dest="surr", type=str, default='PC',
                    help="Surrogate type", choices=['PC', 'NN'])
parser.add_argument("-m", "--method", dest="method", type=str, default='bcs',
                    help="Fitting method", choices=['lsq', 'bcs', 'anl'])
parser.add_argument("-c", "--pctype", dest="pctype", type=str, default='LU',
                    help="PC type", choices=['LU', 'HG'])
parser.add_argument("-o", "--order", dest="order", type=int, default=1,
                    help="PC order.")

args = parser.parse_args()

####################################################################
xlabel = '' #r'log$_{10}$ Pressure'
ylabel = '' #'Amplitude'
plot_samfit = True
####################################################################

method = args.method
pctype = args.pctype
order = args.order
trnfactor = args.trnfactor
surr = args.surr

x = np.loadtxt(args.xdata)
y = np.loadtxt(args.ydata)

# tgr = np.linspace(0, 1, 72)
# y=np.sum(x, axis=1)[:, np.newaxis]**2 * tgr**3

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
    xc = np.loadtxt(args.xcond, ndmin=2)
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

klsurr = KLSurr()
klsurr.build(x, y, surr=surr)
klsurr.plot_parity_xi()

ypred=klsurr.function(x)

#ykl=klsurr.kl.eval(klsurr.kl.xi).T


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
        plt.plot(xc, ypred[isam,:], 'go-', ms=8, label='KL+Surr. Apprx.')
        plt.title(f'Sample #{isam+1}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'fit_s{str(isam).zfill(3)}.png')
        plt.close(f)

# TODO: hasattr is temporary for backward compatibility with elm_glob
if hasattr(klsurr, 'smodel') and isinstance(klsurr.smodel, PCRV):
    mainsens = klsurr.compute_sens_pc()
else:
    mainsens = klsurr.compute_sens(get_domain(x), 1111)

pars = range(ndim)
cases = range(nout)

plot_sens(mainsens,pars,cases,vis="bar",ncol=5,
          par_labels=pnames, case_labels=[str(i) for i in cases],
          lbl_size=25, xticklabel_size=18, legend_size=18,
          figname='sens_klsurr.png')

klsurr.kl.modes_clip(neig_clip=50) # otherwise self.modes is nx x nx, too large

savepk(klsurr, 'klsurr')
