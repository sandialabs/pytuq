#!/usr/bin/env python
"""App to build NN-based surrogates for multioutput models."""
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt


from pytuq.rv.pcrv import PCRV
from pytuq.utils.xutils import read_textlist
from pytuq.utils.plotting import myrc, lighten_color, plot_dm, plot_sens
#from pytuq.utils.maps import scale01ToDom

try:
    from quinn.nns.rnet import RNet, Const, NonPar
except ImportError:
    print("QUiNN not installed. NN fit will not work. Exiting.")
    sys.exit()

torch.set_default_dtype(torch.double)

myrc()


usage_str = 'Script to build NN-based surrogates for multioutput models.'
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

args = parser.parse_args()

####################################################################
xlabel= '' #r'log$_{10}$ Pressure'
ylabel = '' #'Amplitude'
plot_samfit = True

####################################################################

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
assert(len(indtst)>0)


mynn = RNet(111, 3, NonPar(4),
            indim=ndim, outdim=nout, layer_pre=True, layer_post=True, biasorno=True, nonlin=True, mlp=False, final_layer=None)

def lrsched(epoch):
  return 0.001 #np.exp(np.sin(0.002* np.pi*epoch)-5.)

  if epoch>15000:
    return 0.00001
  elif epoch>1000:
    return 0.001
  else:
    return 0.01 #10.0/(epoch+1.)

mynn_best = mynn.fit(x[indtrn,:], y[indtrn, :],
             val=[x[indtst, :], y[indtst, :]],
             lrate=1.0, lmbd=lrsched, batch_size=1000, wd=0.0, nepochs=1000,
             gradcheck=False, freq_out=100, freq_plot=100)


ypred = mynn_best.predict(x)

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
        plt.plot(xc, ypred[isam,:], 'go-', ms=8, label='NN apprx.')
        plt.title(f'Sample #{isam+1}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'fit_s{str(isam).zfill(3)}.png')
        plt.close(f)



