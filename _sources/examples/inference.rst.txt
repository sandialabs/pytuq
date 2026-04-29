====================
Bayesian Inference
====================

Examples demonstrating Bayesian parameter inference via MCMC sampling,
variational inference, and model calibration workflows.


ex_mcmc_banana.py
-----------------

MCMC sampling from a banana-shaped distribution.

Demonstrates different MCMC methods (AMCMC, HMC, MALA) for sampling
from a challenging banana-shaped (Rosenbrock) posterior distribution.


ex_mcmc_fitline.py
------------------

MCMC-based Bayesian linear model calibration.

Uses Adaptive MCMC to calibrate a linear model to noisy data.


ex_mcmc_fitmodel.py
-------------------

MCMC-based multivariate linear model calibration.

Uses Adaptive MCMC to infer parameters of a linear model with multiple
features, including bias and weight parameters.


ex_mfvi.py
----------

Mean-field variational inference.

Uses MFVI with different optimization methods (PSO, Scipy) to approximate
posterior distributions for parameters in a simple model.


ex_minf.py
----------

Model inference workflows.

Shows different approaches to Bayesian parameter inference including
optimization-based and sampling-based methods for model calibration.


ex_minf_sketch.py (:doc:`demo <demo/ex_minf_sketch>`)
-----------------------------------------------------

Comprehensive parameter inference sketch.

Detailed 1D parameter inference example using MCMC.  Demonstrates
configuring likelihood types, prior options, data variance treatment,
and post-processing the calibration results.


ex_evidence.py
--------------

Model selection using Bayesian evidence.

Compares different polynomial chaos models using analytical linear
regression (ANL) and computes evidence values to determine the
best-fitting model order.
