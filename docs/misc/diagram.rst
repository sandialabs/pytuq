===========================
Class Inheritance Diagrams
===========================

The diagrams below highlight the essential inheritance relationships
in PyTUQ — groups where classes form multi-level hierarchies.


Fitting
-------

``fitbase`` is the root fitting class, extended by ``gp`` (Gaussian‑process
fitting) and ``lreg`` (linear regression).

.. inheritance-diagram:: pytuq.fit.fit pytuq.fit.gp pytuq.lreg.lreg
   :top-classes: pytuq.fit.fit.fitbase
   :parts: 1


Linear Regression
-----------------

``lreg`` inherits from ``fitbase`` and is specialised into several
solvers: ``lsq`` (least squares), ``anl`` (analytical Bayesian),
``bcs`` (Bayesian compressive sensing), ``opt`` (optimisation‑based),
and ``lreg_merr`` (model‑error regression).

.. inheritance-diagram:: pytuq.lreg.lreg pytuq.lreg.anl pytuq.lreg.bcs pytuq.lreg.opt pytuq.lreg.merr
   :top-classes: pytuq.lreg.lreg.lreg
   :parts: 1


MCMC Samplers
-------------

``MCMCBase`` is the common ancestor of the three samplers:
``AMCMC`` (adaptive Metropolis), ``HMC`` (Hamiltonian Monte Carlo),
and ``MALA`` (Metropolis‑Adjusted Langevin Algorithm).

.. inheritance-diagram:: pytuq.minf.calib pytuq.minf.mcmc
   :top-classes: pytuq.minf.calib.MCMCBase
   :parts: 1


Priors
------

``Prior`` is specialised into ``Prior_uniform`` and ``Prior_normal``.

.. inheritance-diagram:: pytuq.minf.priors
   :top-classes: pytuq.minf.priors.Prior
   :parts: 1


Likelihoods
-----------

``Likelihood`` branches into several concrete likelihood models.

.. inheritance-diagram:: pytuq.minf.likelihoods
   :top-classes: pytuq.minf.likelihoods.Likelihood
   :parts: 1


Random Variables
----------------

``MRV`` is the multivariate random variable base class, extended by
``GMM``, ``Mixture``, ``Inverse``, and several 1‑D distributions.
``PCRV`` (polynomial‑chaos random variable) has its own sub‑hierarchy.

.. inheritance-diagram:: pytuq.rv.mrv pytuq.rv.pcrv
   :top-classes: pytuq.rv.mrv.MRV
   :parts: 1


Functions
---------

``Function`` is the base for all test functions, operators, and
benchmark functions in PyTUQ.

.. inheritance-diagram:: pytuq.func.func pytuq.func.poly pytuq.func.genz pytuq.func.oper pytuq.func.toy
   :top-classes: pytuq.func.func.Function
   :parts: 1


Global Sensitivity Analysis
----------------------------

``SensMethod`` is the root for all GSA methods.

.. inheritance-diagram:: pytuq.gsa.gsa
   :top-classes: pytuq.gsa.gsa.SensMethod
   :parts: 1


Linear Dimensionality Reduction
--------------------------------

``LinRed`` is extended by ``KLE`` (Karhunen–Loève) and ``SVD``.

.. inheritance-diagram:: pytuq.linred.linred pytuq.linred.kle pytuq.linred.svd
   :top-classes: pytuq.linred.linred.LinRed
   :parts: 1


Optimisers
----------

``OptBase`` is the optimiser base class, with gradient‑descent variants
(``GD``, ``SGD``, ``Adam``), particle‑swarm (``PSO``), and a SciPy wrapper.

.. inheritance-diagram:: pytuq.optim.optim pytuq.optim.gd pytuq.optim.pso pytuq.optim.sciwrap
   :top-classes: pytuq.optim.optim.OptBase
   :parts: 1
