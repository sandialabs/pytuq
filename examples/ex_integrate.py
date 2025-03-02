#!/usr/bin/env python

import numpy as np
from pytuq.utils.integr import *

def single_gaussian(x, w=1.0):

    return np.exp(-0.5*np.sum((x/w)**2, axis=1))

def double_gaussian(x):
    shift=11.0
    y1 = np.exp(-np.sum(x**2, axis=1))
    y2 = 0.2 * np.exp(-0.25*np.sum((x-shift)**2, axis=1))

    return y1 + y2



dim = 3

w = 0.8 #1.202
myfunc = single_gaussian
func_args = {'w': w}
int_exact = w**dim * np.sqrt((2.*np.pi)**dim)

# w = None
# myfunc = double_gaussian
# func_args = {}
# int_exact = np.sqrt(np.pi**dim)+0.2*(2**dim)*np.sqrt(np.pi**dim)

domain = np.tile(np.array([-np.inf,np.inf]), (dim,1))
intg = IntegratorMCMC()
int_mcmc, results = intg.integrate(myfunc,
                                  func_args=func_args,
                                  domain=domain,
                                  nmc=10000)
print("MCMC  estimate  : ", int_mcmc)

# intg = IntegratorWMC()
# int_wmc, results = intg.integrate(myfunc,
#                                   func_args=func_args,
#                                   mean=np.zeros(dim),
#                                   nmc=1000000)
# print(int_wmc)

intg = IntegratorGMM()
int_gmm, results = intg.integrate(myfunc,
                                  func_args=func_args,
                                  means=[np.zeros(dim), 11.*np.ones(dim)],
                                  covs=[0.5*np.eye(dim), 2.0*np.eye(dim)],
                                  nmc=1000000)
print("GMM   estimate  : ", int_gmm)


# intg = IntegratorGMMT()
# int_gmmt, results = intg.integrate(myfunc, np.tile(np.array([-15., 15.]), (dim,1)),
#                                   func_args=func_args,
#                                   means=[np.zeros(dim), 11.*np.ones(dim)],
#                                   covs=[0.5*np.eye(dim), 2.0*np.eye(dim)],
#                                   nmc=1000000)
# print("GMMT  estimate  : ", int_gmmt)

intg = IntegratorMC()
int_mc, results = intg.integrate(myfunc,
                                 domain=np.tile(np.array([-15.0,15.0]), (dim,1)),
                                 nmc=10000000, func_args=func_args)
print("MC    estimate  : ", int_mc)



# domain = np.tile(np.array([-np.inf,np.inf]), (dim,1))
# intg = IntegratorScipy()
# int_nquad, results = intg.integrate(myfunc, domain, func_args=func_args, epsrel=0.001)
# print(int_nquad, results)

print("Exact integral  : ", int_exact)
