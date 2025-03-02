#!/usr/bin/env python

"""[summary]

[description]
"""
import numpy as np
import matplotlib.pyplot as plt


from pytuq.rv.pcrv import PCRV
from pytuq.utils.mindex import get_mi
from pytuq.utils.plotting import myrc, lighten_color, plot_dm, plot_vars
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

N = 11
order = 5
true_model = fcb.sin4 #lambda x: x[:,0]**5 #fcb.sin4
dim = 1

########################################################
########################################################
########################################################



domain = np.ones((dim, 2))*np.array([-1.,1.])

x = scale01ToDom(np.random.rand(N,dim), domain)
y = true_model(x)

pcrv = PCRV(1, dim, 'LU', mi=get_mi(order, dim))

def basisevaluator(x, pars):
    pcrv, = pars
    return pcrv.evalBases(x, 0)


lreg = anl()
#lreg = lsq()
#lreg = opt()

lreg.setBasisEvaluator(basisevaluator, (pcrv,))
lreg.fit(x, y)
print(lreg.cf)


y_pred, y_std = lreg.predict_wstd(x)


plot_dm([y], [y_pred],
        errorbars=[[y_std,y_std]],
        labels=['Training'], colors=None,
        axes_labels=['Data', 'GP Fit'],
        figname='dm.png', msize=4)

if dim==1:
    ngr = 111
    xgrid = np.linspace(domain[0,0], domain[0,1], ngr)[:, np.newaxis]
    ygrid = true_model(xgrid)
    ygrid_pred, ygrid_predstd = lreg.predict_wstd(xgrid)

    plt.figure(figsize=(12,9))
    plt.plot(x[:,0], y, 'ko', ms=11, label='Data', zorder=1000)
    plt.plot(xgrid[:,0], ygrid, 'k--', label='True model', zorder=1000)
    plot_vars(xgrid[:,0], ygrid_pred,
              variances=ygrid_predstd[:,np.newaxis]**2,
              varlabels=['Fit stdev'],
              varcolors=[lighten_color('m', 0.5)],
              ax=plt.gca())

    plt.legend()
    plt.savefig('fit.png')
    plt.clf()
