#!/usr/bin/env python

"""[summary]

[description]
"""
import numpy as np
import matplotlib.pyplot as plt


from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi
from pytuq.utils.plotting import myrc, lighten_color, plot_dm
from pytuq.utils.maps import scale01ToDom
from pytuq.lreg.merr import lreg_merr
from pytuq.lreg.anl import anl
from pytuq.lreg.opt import opt
from pytuq.lreg.lreg import lsq
import pytuq.utils.funcbank as fcb

myrc()

########################################################
########################################################
########################################################

N = 14
order = 4
true_model = fcb.sin4
dim = 1

########################################################
########################################################
########################################################

# Domain of input
domain = np.ones((dim, 2))*np.array([-1.,1.])

# Generate x-y data
x = scale01ToDom(np.random.rand(N,dim), domain)
y = true_model(x)


# Create A-matrix (based on PC basis)
pcrv = PCRV(1, dim, 'LU', mi=get_mi(order, dim))
Amat = pcrv.evalBases(x, 0)

# Pick a method for regression
lreg = anl()
#lreg = anl(method='vi')
#lreg = lsq()
#lreg = opt()

# Fit the data
lreg.fita(Amat, y)
# Predict
y_pred, y_pred_var, _ = lreg.predicta(Amat, msc=1)

# Plot diagonal parity plot
plot_dm([y], [y_pred],
        errorbars=[[np.sqrt(y_pred_var),np.sqrt(y_pred_var)]],
        labels=['Training'], colors=None,
        axes_labels=['Data', 'GP Fit'],
        figname='dm.png', msize=6)


# For 1d problems, plot the actual fit function
if dim==1:
    xg = np.linspace(-1., 1., 100).reshape(-1, 1)
    yg = true_model(xg)
    Agr = pcrv.evalBases(xg, 0)
    ygr_pred, ygr_pred_var, _ = lreg.predicta(Agr, msc=1)

    plt.plot(x, y, 'ko', ms=11, label='Data')
    plt.plot(xg, yg, 'k--', label='True model')
    ygr_pred_std = np.sqrt(ygr_pred_var)

    p, = plt.plot(xg[:, 0], ygr_pred, 'm-', linewidth=5, label='Fit mean')
    lc = lighten_color(p.get_color(), 0.5)
    plt.fill_between(xg[:, 0],
                     ygr_pred - ygr_pred_std,
                     ygr_pred + ygr_pred_std,
                     color=lc, zorder=-1000, alpha=1.0,
                     label='Fit stdev')
    plt.legend()
    plt.savefig('fit.png')
    plt.clf()
