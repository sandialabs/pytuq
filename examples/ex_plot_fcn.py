#!/usr/bin/env python
"""Example demonstrating anchored 1D and 2D function plotting utilities.

This script shows how to use plotting utilities to visualize model functions
with respect to one or two parameters while fixing others at nominal values.
"""

import numpy as np

from pytuq.utils.plotting import myrc, plot_1d_anchored, plot_2d_anchored

myrc()



# from quinn.nns.nnwrap import nn_p
# from .myutils import tch
# def nnloss(p, modelpars):
#     nnmodel, loss_fn, xtrn, ytrn = modelpars

#     npt = p.shape[0]
#     fval = np.empty((npt,))
#     for ip, pp in enumerate(p):
#         fval[ip]=loss_fn(tch(nn_p(pp, xtrn, nnmodel).reshape(ytrn.shape)), tch(ytrn)).item()

#     #print(fval)

#     return fval


def model1(p, modelpars):
    return np.mean(np.exp(p), axis=1)

def model2(p, modelpars):
    return -np.mean(np.exp(p), axis=1)

def model3(p, modelpars):
    return -np.mean(np.sin(10.*p), axis=1)






models = [model1, model2, model3]
modelpars=[None, None, None]
labels=['Model1', 'Model2', 'Model3']

pdim = 5

# plot_fcn1d(models, modelpars, ndim=pdim,
#            ngr=111, ncol=8, nrow=5, labels=labels)
plot_1d_anchored(models, modelpars, np.random.rand(pdim,),
                 scale=1., modellabels=labels, legend_show=True,
                 clearax=True, ncolrow=(8,5))

# plot_fcn2d(models, modelpars, ndim=pdim,
#            ngr=33, ncol=5, nrow=5, labels=labels)

plot_2d_anchored(models, modelpars, np.random.rand(pdim,),
                 scale=1., modellabels=labels, legend_show=True,
                 clearax=True, ncolrow=(5,5))

