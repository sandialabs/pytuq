==================
Polynomial Chaos
==================

Examples demonstrating polynomial chaos (PC) basis construction,
random variable operations, multiindex manipulation, quadrature,
uncertainty propagation, and model selection.


ex_pcbasis1d.py
---------------

1D polynomial chaos basis evaluation and plotting.

Evaluates and plots Hermite polynomial basis functions of various orders
to illustrate orthogonal polynomial behavior.


ex_pcrv.py
----------

Polynomial chaos random variable slicing.

Shows how to slice a PC random variable by fixing certain dimensions
at nominal values to obtain a reduced-dimension PCRV.


ex_pcrv1.py
-----------

PCRV compression and random dimension selection.

Creates a multivariate normal PCRV with specified random dimensions,
samples from it, and demonstrates PC compression operations.


ex_pcrv2.py
-----------

Basic polynomial chaos random variable operations.

Creates a PCRV with random coefficients and demonstrates computing
statistics (mean, variance), basis norms, and sampling.


ex_pcrv_mvn.py
--------------

Multivariate normal polynomial chaos random variables.

Creates ``PCRV_mvn`` objects with specified means and covariances,
and generates samples from the multivariate normal distribution.


ex_mrv.py
---------

Multivariate random variable (MRV) operations.

Shows how to create and manipulate polynomial chaos random variables
including independent and multivariate normal PC random variables.


ex_mindex.py
------------

Multiindex generation and encoding.

Shows how to generate polynomial chaos multiindices and encode them
for efficient storage and manipulation.


ex_quad.py
----------

Quadrature point generation for PC germ variables.

Generates and visualizes quadrature points for polynomial chaos germ
variables using tensor product quadrature rules.


ex_uprop.py
-----------

Uncertainty propagation through a model with PC inputs.

Shows how to propagate polynomial chaos input uncertainties through
a nonlinear model using projection or regression methods.


ex_uprop2.py
------------

Uncertainty propagation via projection and regression.

Compares projection-based and regression-based methods for propagating
PC input uncertainties through nonlinear forward models.
