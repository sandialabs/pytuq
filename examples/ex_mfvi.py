#!/usr/bin/env python
"""Example demonstrating Mean-Field Variational Inference (MFVI) for Bayesian inference.

This script uses MFVI with different optimization methods (PSO, Scipy) to approximate
posterior distributions for parameters in a simple model.
"""

import numpy as np
#from autograd import grad

import matplotlib.pyplot as plt


from pytuq.minf.vi import MFVI
from pytuq.optim.pso import PSO

from pytuq.func.func import Function
from pytuq.func.bench1d import TFData
from pytuq.utils.maps import scale01ToDom
from pytuq.utils.plotting import plot_dm, myrc

myrc()


#########################################################################
#########################################################################
#########################################################################

domain = np.array([[-20., 60.]])

ngr = 133
ntrn = 150

xtrn = scale01ToDom(np.random.rand(ntrn), domain)[:, np.newaxis]

fcn = TFData()

ytrn = fcn(xtrn) + np.random.randn(ntrn, 1)

ytrn = (ytrn - ytrn.mean()) / ytrn.std()

print(xtrn.shape, ytrn.shape)

pdim = 3


class linear_model(Function):
    def __init__(self, xdata):
        super().__init__()
        self.xdata = xdata

    def __call__(self, p):
        """y=ax+b
        """
        a = p[:, 0]
        b = p[:, 1]

        return np.outer(a, self.xdata) + b.reshape(-1, 1)


xgrid = np.linspace(domain[0, 0], domain[0, 1], ngr)
# modelpar_true = (3, -2)
# ygrid_true = linear_model(xgrid, modelpar_true[0], modelpar_true[1])


# Set the initial parameters and run Optimization
nsteps = 10000  # number of steps
param_ini = np.random.rand(2 * pdim)  # initial parameter values

lossinfo = {'nmc': 222, 'datasigma': 1.0}
mfvi = MFVI(linear_model(xtrn), ytrn[:,0], pdim, lossinfo, reparam='logexp')
objective, objectivegrad, objectiveinfo = mfvi.eval_loss, mfvi.eval_loss_grad_, {}


myopt = PSO(2*pdim)
myopt.setObjective(objective, objectivegrad, **objectiveinfo)
results = myopt.run(100, np.random.rand(2*pdim,))
cmode, pmode = results['best'], results['bestobj']




print(cmode, pmode)


# Postprocess

mfvi.plot_parpdf(cmode)

nsam = 1000
ypred_all = mfvi.compute_pred(cmode, nsam)
q1, ypred_med, q2 = np.quantile(ypred_all, [0.05, 0.5, 0.95], axis=0)
plot_dm([ytrn], [ypred_med],
        errorbars=[(ypred_med-q1, q2-ypred_med)],
        labels=['Training'], colors=['b'],
        axes_labels=['Data', 'Model'], figname='dm_mfvi.png',
        legendpos='in', msize=8)


# For 1d x-values, plot the fit
grid_model = linear_model(xgrid)
ypred_grid_all = mfvi.compute_pred(cmode, nsam, grid_model)
q1, ygrid_pred_med, q2 = np.quantile(ypred_grid_all, [0.05, 0.5, 0.95], axis=0)

fig = plt.figure(figsize=(12,8))
for ypred in ypred_grid_all:
    plt.plot(xgrid, ypred, '-', color='grey', lw=0.1)
plt.fill_between(xgrid, q1, q2, color='b', alpha=0.4, label='90% quantile', zorder=-10000)

ypred_grid_mode = grid_model(cmode.reshape(1, -1))[0]
plt.plot(xgrid, ypred_grid_mode, 'b-', label='Best fit', lw=3)
plt.plot(xtrn, ytrn, 'k*', label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('posterior_fit.png')
