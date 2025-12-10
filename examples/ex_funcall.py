#!/usr/bin/env python
"""Example demonstrating automatic instantiation and testing of all PyTUQ function classes.

This script automatically creates instances of all available function classes from
PyTUQ's function modules and validates their gradient implementations.
"""

import numpy as np

from pytuq.utils.plotting import myrc
from pytuq.utils.xutils import instantiate_classes_from_module
myrc()

objects = []
for submod in ['bench', 'bench1d', 'bench2d', 'benchNd', 'chem', 'genz', 'poly', 'toy']:
    this_objects = instantiate_classes_from_module(f"pytuq.func.{submod}")
    for j in this_objects:
        if j.name not in ['GenzBase', 'Poly']:
            objects.append(j)

#print("Created instances:", objects)
for fcn in objects:

    print(f"========== Function {fcn.name} ==================")
    print(fcn.name, "->", fcn)


    print("Gradient check")
    x = np.random.rand(111, fcn.dim)
    assert(np.allclose(fcn.grad_(x, eps=1.e-8), fcn.grad(x), atol=1.e-5, rtol=1.e-3))

    # print("Minimize")
    # xmin = fcn.minimize()
    # print(f"Minimum is at {xmin}")

    print(f"Domain is {fcn.domain}")

    nom = fcn.sample_uniform(1)[0]
    print("Plotting 1d slice")
    fcn.plot_1d(ngr=100, nom=nom)

    if fcn.dim>1:
        print("Plotting 2d slice")
        fcn.plot_2d(ngr=52, nom=nom)
