=========================
Regression & Surrogates
=========================

Examples demonstrating surrogate model construction via linear regression,
Bayesian compressive sensing, Gaussian processes, and dimensionality
reduction.


ex_lreg.py
----------

Polynomial chaos linear regression methods.

Compares various linear regression techniques (LSQ, ANL, OPT, MERR)
for constructing polynomial chaos surrogates and evaluates their
performance.


ex_lreg_basiseval.py
--------------------

Polynomial chaos basis evaluation and regression.

Compares different regression methods (least squares, analytical,
optimization) for polynomial chaos surrogate construction and evaluates
basis function efficiency.


ex_lreg_merr.py
---------------

Linear regression with measurement error.

Performs polynomial chaos regression accounting for measurement errors
in the data using the MERR (Measurement Error in Regression) method.


ex_bcs.py (:doc:`demo <demo/ex_bcs>`)
-----------------------------------------

Bayesian compressive sensing for sparse PC regression.

Uses BCS to construct a sparse polynomial chaos surrogate model with a
specified multiindex and model data, comparing predictions with true
function values.


ex_bcs_mindex_growth.py
-----------------------

Adaptive multiindex growth with BCS.

Iteratively grows a polynomial chaos surrogate using adaptive multiindex
selection and BCS regression for sparse approximation.


ex_gp.py
--------

Gaussian process regression.

Builds a Gaussian process surrogate model from training data, performs
hyperparameter optimization, and evaluates prediction accuracy.


ex_kl.py
--------

Karhunen--Loève expansion and SVD.

Builds KLE or SVD representations of model output data to capture
variance with reduced dimensionality.


ex_klpc.py
----------

KLE combined with polynomial chaos.

Uses Karhunen--Loève Expansion to reduce output dimensionality, then
builds PC surrogates for the reduced modes to efficiently represent
high-dimensional model outputs.
