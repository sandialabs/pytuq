#!/usr/bin/env python

"""
Python Toolkit for Uncertainty Quantification (PyTUQ)
======================================================
 
PyTUQ is a comprehensive Python toolkit for uncertainty quantification (UQ)
in computational models. It provides a wide range of methods and tools for
quantifying, propagating, and analyzing uncertainties in numerical simulations.
 
Main Capabilities
-----------------
- **Polynomial Chaos Expansions (PCE)**: Construct surrogate models using
  orthogonal polynomial bases for efficient uncertainty propagation
- **Bayesian Inference & MCMC**: Perform parameter calibration and inference
  using various Markov Chain Monte Carlo methods
- **Global Sensitivity Analysis**: Compute Sobol indices and other sensitivity
  metrics to identify important parameters
- **Gaussian Process Regression**: Build probabilistic surrogate models with
  uncertainty quantification
- **Dimensionality Reduction**: Apply SVD and Karhunen-Loeve expansions to
  reduce problem complexity
- **Bayesian Compressive Sensing (BCS)**: Perform sparse regression to identify
  relevant polynomial terms automatically
- **Linear Regression Methods**: Various techniques for model fitting including
  ordinary least squares, ridge regression, and more
- **Neural Network Surrogates**: Construct surrogate models using neural networks
- **Random Variable Classes**: Handle multivariate random variables with various
  probability distributions
 
Module Organization
-------------------
pytuq.pce
    Polynomial chaos expansion classes and utilities
pytuq.mcmc
    Markov Chain Monte Carlo methods for Bayesian inference
pytuq.gsa
    Global sensitivity analysis tools
pytuq.gp
    Gaussian process regression
pytuq.dr
    Dimensionality reduction techniques
pytuq.lreg
    Linear regression and Bayesian compressive sensing
pytuq.nn
    Neural network surrogate models
pytuq.rv
    Random variable classes and transformations
pytuq.utils
    Utility functions, test functions, and integration methods
 
Quick Start Example
-------------------
Here's a simple example using polynomial chaos expansion::
 
    import numpy as np
    from pytuq.rv.pcrv import PCRV
    from pytuq.utils.mindex import get_mi
 
    # Define problem dimension and polynomial order
    dim = 2
    order = 3
 
    # Create multiindex for polynomial basis
    mindex = get_mi(order, dim)
 
    # Create PC random variable with uniform distributions
    pcrv = PCRV(1, dim, 'LU', mi=mindex)
 
    # Generate random samples
    samples = np.random.rand(100, dim)
 
    # Evaluate polynomial bases
    basis_values = pcrv.evalBases(samples, 0)
 
For More Information
--------------------
- Documentation: https://sandialabs.github.io/pytuq/
- Source Code: https://github.com/sandialabs/pytuq
- Installation: See the installation guide in the documentation
- Examples: Explore the examples/ directory for detailed use cases
 
License
-------
Distributed under BSD 3-Clause License. See LICENSE.txt for more information.
"""
 

