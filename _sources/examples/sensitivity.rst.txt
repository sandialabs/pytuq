=======================
Sensitivity Analysis
=======================

Examples demonstrating global sensitivity analysis using sampling-based
and PC-based Sobol indices.


ex_gsa.py
---------

Global sensitivity analysis using Sobol indices.

Computes main and total Sobol sensitivity indices using either
sampling-based (SamSobol) or PC-based (PCSobol) methods for a test
function.


ex_gsa_multi.py
---------------

Global sensitivity analysis for multi-output models.

Performs Sobol sensitivity analysis on a simple multi-output model to
compute main and total sensitivity indices for each output dimension.


ex_pcgsa.py
-----------

PC-based global sensitivity analysis.

Computes Sobol sensitivity indices using polynomial chaos expansions,
optionally building a PC surrogate first or using the Ishigami function
directly.
