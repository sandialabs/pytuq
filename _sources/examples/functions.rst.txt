============================
Functions & Optimization
============================

Examples demonstrating function construction, gradient checking,
benchmark functions, optimization algorithms, and numerical integration.


ex_func.py
----------

Function composition and operations.

Shows how to combine, transform, and manipulate various function objects
including toy functions, Genz functions, chemistry functions, and
benchmark functions.


ex_funcall.py
-------------

Automatic testing of all function classes.

Automatically creates instances of all available function classes from
PyTUQ's function modules and validates their gradient implementations.


ex_funcgrad.py
--------------

Gradient checking and evaluation.

Tests analytical gradients of benchmark functions against numerical
gradients, and visualizes function values and derivatives.


ex_genz1d.py
------------

1D Genz test functions.

Evaluates and plots various 1D Genz functions including oscillatory,
corner peak, and sum functions across their domains.


ex_optim.py
-----------

Optimization algorithms on the Rosenbrock function.

Compares different optimization methods (Gradient Descent, Adam, PSO,
Scipy) for minimizing the Rosenbrock function.


ex_integrate.py
---------------

Numerical integration of Gaussian functions.

Tests various integration methods (MCMC, GMM, MC) on single and double
Gaussian functions, comparing numerical results with analytical
solutions.


ex_orf.py
---------

Orthonormalization of functions.

Tests orthonormalization of functions using Gram-Schmidt or QR
decomposition.  Written by Habib N. Najm (2025).
