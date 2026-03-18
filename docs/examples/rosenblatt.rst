============================
Rosenblatt Transformation
============================

Examples demonstrating the Rosenblatt transformation for mapping between
arbitrary distributions and uniform samples, and constructing PC
representations from such mappings.


ex_ros.py
---------

Basic Rosenblatt transformation in 2D.

Demonstrates forward and inverse Rosenblatt transformations.  Constructs a
Rosenblatt map from exponential-uniform samples, verifies that the forward
map produces uniform samples, and that the inverse resamples from the
original distribution.


ex_iros_1d.py
-------------

1D forward and inverse Rosenblatt transformation.

Constructs a Rosenblatt map from samples of an exponential-uniform
distribution and plots both the forward and inverse maps against the true
transformation.


ex_iros_2d.py
-------------

2D forward and inverse Rosenblatt transformation.

Constructs a Rosenblatt map from samples drawn from an exponential-uniform
distribution and plots the conditional mapping slices for each dimension.


ex_ros_pc.py
------------

PC from Rosenblatt transformation.

Maps samples to uniform via the Rosenblatt map, then fits PC coefficients
to the inverse map using analytical regression.


ex_ros_pcj.py
-------------

Joint PC from Rosenblatt transformation.

Constructs a polynomial chaos representation from samples using a joint
regression approach in the combined parametric and stochastic space.


ex_ros_pcs.py
-------------

Per-sample PC from Rosenblatt transformation.

Builds independent PC fits per parameter sample, contrasting with the
joint approach in ``ex_ros_pcj.py``.


ex_rospc_multiple.py
--------------------

Repeated PC from Rosenblatt tests.

Repeatedly builds PC representations from random samples using the
Rosenblatt transformation to assess consistency of the estimated mean and
standard deviation across replicas.
