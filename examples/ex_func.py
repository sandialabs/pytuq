#!/usr/bin/env python
"""Example demonstrating function composition and operations with PyTUQ function classes.

This script shows how to combine, transform, and manipulate various function objects
including toy functions, Genz functions, chemistry functions, and benchmark functions.
"""

import numpy as np

from pytuq.utils.mindex import get_mi, get_npc
from pytuq.func import toy, genz, chem, bench2d, bench, benchNd, poly, oper, func
from pytuq.utils.plotting import myrc

myrc()

fcns = [
        func.ModelWrapperFcn(lambda x,p : x[:,0]**p[0]+np.sin(x[:,1]**p[0]), 3, modelpar=[3]), \
        oper.PickDim(2, 1, cf=100.)+chem.MullerBrown(),\
        oper.PickDim(2, 1, cf=1.)-chem.MullerBrown(),\
        bench2d.Adjiman()*bench2d.Branin(), \
        oper.PickDim(2, 1, cf=1.) / (toy.Constant(2,np.ones(1,)) + oper.PickDim(2, 0, cf=1.)), \
        oper.PickDim(2, 1, cf=100.)**3, \
        toy.Quad(),\
        toy.Quad2d(),\
        toy.Exp(weights=[2., -1.]),\
        toy.Log(),\
        toy.Constant(2, np.ones(2,)),\
        toy.Identity(2),\
        genz.GenzOscillatory(shift=1., weights=[5., 2.]),\
        genz.GenzSum(shift=-1., weights=[3., 1.]),\
        genz.GenzCornerPeak(weights=[7., 2.]),\
        chem.MullerBrown(),\
        chem.LennardJones(),\
        benchNd.Sobol(dim=3),\
        bench.Ishigami(),\
        benchNd.NegAlpineN2(),\
        benchNd.SumSquares(),\
        benchNd.MVN([-1., 2.], [[2., -1.], [-1., 1.]]),\
        poly.Leg(get_mi(4,3), np.ones((get_npc(4, 3),))),\
        poly.Mon(get_mi(4,3), np.ones((get_npc(4, 3),))),\
        oper.CartesProdFcn(toy.Identity(1),toy.Identity(1)), \
        oper.LinTransformFcn(toy.Identity(2), 3., -2.),\
        oper.ShiftFcn(chem.MullerBrown(), [-0.3,0.3]),\
        oper.SliceFcn(chem.MullerBrown(), ind=[0,1]),\
        oper.ComposeFcn(toy.Identity(2), genz.GenzOscillatory(weights=[-3., -1.])),\
        oper.ComposeFcn(toy.Exp(), genz.GenzOscillatory(), name='Composite1d'),\
        oper.GradFcn(bench2d.Adjiman(), 1), \
        oper.GradFcn(bench2d.Franke(), 1), \
        oper.PickDim(2, 1)
        ]


for fcn in fcns:
    print(f"========== Function {fcn.name} ==================")
    print("Gradient check")
    x = np.random.rand(111, fcn.dim)
    assert(np.allclose(fcn.grad_(x, eps=1.e-8), fcn.grad(x), atol=1.e-5, rtol=1.e-3))

    print("Minimize")
    xmin = fcn.minimize()
    print(f"Minimum is at {xmin}")

    print(f"Domain is {fcn.domain}")

    print("Plotting 1d slice")
    fcn.plot_1d(ngr=556)

    if fcn.dim>1:
        print("Plotting 2d slice")
        fcn.plot_2d(ngr=55)

