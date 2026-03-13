==========
Forward UQ
==========

The scripts in ``apps/uqpc/`` propagate input parameter uncertainties to
model outputs via Polynomial Chaos (PC) expansions.  As the most common
use-case, they construct a PC surrogate for a multi-output computational
model treated as a black-box simulation code, and then perform global
sensitivity analysis of the outputs with respect to the input parameters.

The folder contains:

- ``uq_pc.py`` — the main driver script.
- ``plot.py`` — post-processing visualisations.
- ``model.x`` — an example black-box model (awk script).
- ``workflow_uqpc.x`` — an example end-to-end workflow shell script.


Workflow overview
-----------------

A typical UQPC session follows five steps:

1. **Setup inputs** — Define uncertain input parameters by specifying
   marginal PCs, a parameter domain, or samples.  Use ``pc_prep.py``
   (see :doc:`pc`) to produce the input PC coefficient file ``pcf.txt``.

2. **Generate samples** — Draw training (and optionally testing)
   realisations from the input PC.  In *online* regimes this is done
   automatically; in *offline* mode the user supplies ``ptrain.txt``,
   ``qtrain.txt``, ``ytrain.txt`` (and test counterparts) beforehand.

3. **Evaluate the model** — Either call a built-in benchmark
   (``online_example``), execute a user-supplied black-box executable
   ``model.x`` (``online_bb``), or load pre-computed outputs
   (``offline``).

4. **Build the PC surrogate** — Fit an output PC expansion via
   analytical projection (``anl``), least-squares (``lsq``), or
   Bayesian Compressed Sensing (``bcs``), and compute Sobol sensitivity
   indices.

5. **Post-process** — Visualise results with ``plot.py``.

The example workflow ``workflow_uqpc.x`` demonstrates all five steps with
commented alternatives.


uq_pc.py
---------

Main driver for the forward-UQ pipeline.

``uq_pc.py`` supports three run regimes:

* ``online_example`` — uses a built-in benchmark function (Ishigami by
  default) defined inside the script.
* ``online_bb`` — calls a user-supplied black-box executable ``model.x``
  that maps an input file to an output file.
* ``offline`` — reads pre-computed training (and testing) data from
  ``ptrain.txt``, ``qtrain.txt``, ``ytrain.txt`` (and their test
  counterparts).

The script constructs input PC representations, generates or reads
training/testing samples, fits a PC surrogate, computes Sobol
sensitivity indices, and bundles everything into ``results.pk``.

**Outputs:**

- ``results.pk`` — Pickled dictionary with the surrogate (``pcrv``),
  training/testing data, and sensitivity indices.
- ``ptrain.txt``, ``qtrain.txt``, ``ytrain.txt`` — Training data
  (parameters, PC germs, model outputs).
- ``ptest.txt``, ``qtest.txt``, ``ytest.txt`` — Testing data (when
  ``-v > 0``).

**Examples:**

.. code-block:: bash

   # Built-in Ishigami example, analytical projection, 7-pt quadrature
   python uq_pc.py -r online_example -t 3 -n 7 -m anl

   # Black-box model, random sampling, BCS fit
   python uq_pc.py -r online_bb -c pcf.txt -x LU -d 3 -o 1 -m bcs -s rand -n 100 -v 30 -t 4

   # Offline data, least-squares fit
   python uq_pc.py -r offline -c pcf.txt -x HG -d 3 -o 1 -m lsq -n 111 -v 33 -t 3

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - ``-r, --regime``
     - ``online_example``
     - Run regime: ``online_example``, ``online_bb``, or ``offline``.
   * - ``-p, --pdom``
     - ``None``
     - Parameter domain file (two columns: lower, upper bound per row).
       If given, ``-o`` and ``-x`` are overwritten to ``1`` and ``LU``.
   * - ``-c, --pcfile``
     - ``None``
     - Input PC coefficient file (e.g. ``pcf.txt`` from ``pc_prep.py``).
   * - ``-d, --pcdim``
     - ``None``
     - Stochastic dimensionality of the input PC.  Required with ``-c``.
   * - ``-x, --pctype``
     - ``LU``
     - PC basis type (``LU``, ``HG``, ``LU_N``, ``LG``, ``JB``, ``SW``).
   * - ``-o, --pcord``
     - ``1``
     - Input PC order.
   * - ``-m, --method``
     - ``anl``
     - Fitting method: ``anl`` (analytical/projection), ``lsq``
       (least-squares), or ``bcs`` (Bayesian Compressed Sensing).
   * - ``-s, --sampl``
     - ``quad``
     - Sampling method: ``quad`` (quadrature) or ``rand`` (random).
   * - ``-n, --nqd``
     - ``7``
     - Number of quadrature points per dimension (``quad``) or total
       number of training points (``rand``).
   * - ``-v, --ntst``
     - ``0``
     - Number of testing points (0 = no testing).
   * - ``-t, --outord``
     - ``3``
     - Output PC order.
   * - ``-e, --tol``
     - ``1e-3``
     - Tolerance for BCS (only used when ``-m bcs``).
   * - ``-z, --seed``
     - ``None``
     - Random seed for reproducibility.

.. note::

   Exactly one of ``-p`` (domain file), ``-c`` (PC coefficient file), or
   ``-d`` (dimensionality) should be provided.  If none are given, a
   2-dimensional default is used.  Providing both ``-p`` and ``-c``, or
   both ``-p`` and ``-d``, is an error.


model.x
-------

An example black-box model supplied as a Bash/AWK script.

It reads an input matrix file and writes an output matrix file::

    ./model.x ptrain.txt ytrain.txt

The default implementation maps 3 inputs to 5 outputs.  Users should
replace this with their own simulation code.


plot.py
-------

Visualise results stored in ``results.pk`` after surrogate construction.

Parameter and output names are read from ``pnames.txt`` and
``outnames.txt`` when present; otherwise generic names are used.

**Examples:**

.. code-block:: bash

   python plot.py sens main            # main Sobol sensitivity bar chart
   python plot.py sens total           # total Sobol sensitivity bar chart
   python plot.py dm training testing  # model-vs-surrogate parity plots
   python plot.py fit training         # per-sample fit overlays
   python plot.py 1d training          # 1-D surrogate slices with training data
   python plot.py 2d                   # 2-D surrogate contour plots
   python plot.py pdf                  # output PDFs
   python plot.py joy                  # joy (ridge-line) plots
   python plot.py jsens                # circular joint-sensitivity plot
   python plot.py sensmat main         # sensitivity heat-map matrix

**Sub-commands:**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Command
     - Description
   * - ``sens``
     - Bar chart of Sobol sensitivity indices.  Takes a second argument
       ``main`` or ``total``.
   * - ``jsens``
     - Circular joint-sensitivity plots for all outputs and their average.
   * - ``sensmat``
     - Sensitivity matrix heatmap for the most important inputs.
       Takes ``main`` or ``total``.
   * - ``dm``
     - Model-vs-surrogate parity (diagonal) plots for each output.
       Takes ``training``, ``testing``, or both.
   * - ``fit``
     - Per-sample model-vs-surrogate overlays.
       Takes ``training``, ``testing``, or both.
   * - ``1d``
     - 1-D surrogate slices along each input dimension (remaining
       parameters at nominal or integrated out).
       Optionally overlay ``training`` and/or ``testing`` data.
   * - ``2d``
     - 2-D surrogate contour plots for all input pairs and outputs.
   * - ``pdf``
     - Probability density plots of the surrogate output distribution.
   * - ``joy``
     - Joy (ridge-line) plots of the surrogate output distribution.


workflow_uqpc.x
---------------

An annotated Bash script that demonstrates the complete workflow.

It shows four alternative ways to define uncertain inputs (normal
marginals, uniform marginals, multivariate normal, or samples), then
walks through sampling, model evaluation, surrogate fitting, data
visualisation, and post-processing.  Users can adapt this script as a
template for their own problems.

Run it from the ``apps/uqpc/`` directory:

.. code-block:: bash

   bash workflow_uqpc.x
