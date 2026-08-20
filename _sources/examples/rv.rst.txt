===========================
Random Variables
===========================

Examples demonstrating random variable construction, distribution
sampling, Gaussian mixture models, and mixture distributions.


ex_gmm.py
---------

Gaussian mixture model sampling and visualization.

Creates a GMM with multiple components, samples from it within a
specified domain, and visualizes the samples and probability density.


ex_sampling.py
--------------

Domain-constrained sampling from GMMs.

Samples from a GMM within a specified domain and visualizes the
resulting samples and probability densities.


ex_mixture.py
-------------

Mixture distributions with Weibull and Gaussian components.

Creates and samples from a mixture distribution combining Weibull and
multivariate normal distributions with specified weights.


ex_webull.py
------------

Weibull distribution and mixture models.

Creates Weibull distributions and mixtures with Gaussian components,
demonstrating sampling from complex mixed distributions.


ex_mcmcrv.py
------------

MCMC random variable.

Demonstrates the ``MCMCRV`` class for defining random variables via
MCMC sampling.  Creates an MCMC-based random variable from a Gaussian
log-posterior, draws samples, and evaluates unscaled PDF values.
