#!/usr/bin/env python
"""Example demonstrating model selection using Bayesian evidence computation.

This script compares different models using analytical linear regression (ANL) and 
computes evidence values to determine which model best fits the data.
"""
import numpy as np
import matplotlib.pyplot as plt


from pytuq.rv.pcrv import PCRV
from pytuq.lreg.anl import anl
from pytuq.utils.mindex import get_mi
from pytuq.utils.maps import scale01ToDom
from pytuq.utils.plotting import myrc, lighten_color

import pytuq.utils.funcbank as fcb

myrc()

########################################################
########################################################
########################################################

def true_model_0(xx):
    y = 0.0*xx[:,0]

    return y

def true_model_1(xx):
    y = 2.*np.sum(xx, axis=1)**4+0.7*xx[:,0]**3-xx[:,0]

    return y


def true_model_2(xx):
    y = 0.7*xx[:,0]
    return y

########################################################
########################################################
########################################################


N = 111
dim = 1
datavar=0.001
true_model = fcb.sin4 #true_model_2 #true_model_1
orders = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

########################################################
########################################################
########################################################

domain = np.ones((dim, 2))*np.array([-1.,1.])
x = scale01ToDom(np.random.rand(N,dim), domain)
y = true_model(x)
y+=np.sqrt(datavar)*np.random.randn(N,)

xg = np.linspace(-1., 1., 100).reshape(-1, 1)
yg = true_model(xg)

ypreds = []
evids = []
for iord, order in enumerate(orders):
    print(f"Fitting order = {order}")

    pcrv = PCRV(1, dim, 'LU', mi=get_mi(order, dim))
    Amat = pcrv.evalBases(x, 0)
    lreg = anl(method='full', prior_var=100., datavar=datavar)
    lreg.fita(Amat, y)
    evidence = lreg.compute_evidence(Amat, y)
    print(f"Evidence = {evidence}")
    evids.append(evidence)

    # Plotting
    Agr = pcrv.evalBases(xg, 0)
    ygr_pred, ygr_pred_var, _ = lreg.predicta(Agr, msc=1)

    ypreds.append(ygr_pred)

    plt.figure(figsize=(10,9))
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
    plt.savefig(f'fit_o{order}.png')
    plt.clf()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.plot(x, y, 'ko', ms=11, markeredgecolor='w', label='Data', zorder=1000)
ax1.plot(xg, yg, 'k--', label='True model', zorder=10000)
for i, ygr_pred in enumerate(ypreds):
    p, = ax1.plot(xg[:, 0], ygr_pred, '-', linewidth=5, label=f'Order {orders[i]} ')
ax1.legend(bbox_to_anchor=(1.0, 1.15), ncol=4, fontsize=14)
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2.set_xlabel('Order')
ax2.set_ylabel('Model Evidence')
ax2.plot(orders, evids, 'o-', ms=10, markeredgecolor='w')

plt.savefig('fit_evid.png')
