==========
Inverse UQ
==========

The scripts in ``apps/iuq/`` perform parameter inference using Markov
chain Monte Carlo (MCMC) sampling with a Polynomial Chaos (PC) surrogate
model constructed by the sibling :doc:`Forward UQ <uqpc>` workflow.
Besides conventional Bayesian inference, embedded model-error inference
is also supported.

The workflow can run out-of-the-box as a demo, but is intended to be
adapted to specific problems — there are many tuning knobs that
typically need adjusting.

The folder contains:

- ``workflow_iuq.x`` — an example end-to-end workflow script.
- ``run_infer.py`` — the main inference script.
- ``create_data_truepc.py`` — synthetic data generation from a PC model.
- ``prep_data.x`` — an alternative example of manual data preparation.
- ``postp_infer.py`` — post-processing and visualisation.


Workflow overview
-----------------

A typical IUQ session follows three steps (after the forward surrogate
has been built with the UQPC workflow):

0. **Prepare data** — Create or supply observational data
   (``ydata.txt``) and the corresponding noise variance
   (``ydatavar.txt``).  Use ``create_data_truepc.py`` for synthetic data
   from the PC surrogate, or ``prep_data.x`` for manual data setup.

1. **Run inference** — Execute ``run_infer.py`` to perform MCMC-based
   Bayesian calibration using the PC surrogate as the forward model.
   Use ``-m`` for embedded model-error inference.

2. **Post-process** — Run ``postp_infer.py`` to generate posterior
   diagnostics: MCMC chain traces, marginal PDFs, prior-vs-posterior
   comparisons, and posterior predictive checks.

The example workflow ``workflow_iuq.x`` demonstrates all three steps.


Prerequisites
-------------

All scripts expect ``results.pk`` (produced by ``uq_pc.py`` in the
:doc:`Forward UQ <uqpc>` workflow) to be present in the working
directory.


create_data_truepc.py
---------------------

Create synthetic observational data for inference testing.

This script loads a previously built PC surrogate from ``results.pk``,
evaluates it at a true parameter vector, adds Gaussian noise, and writes
out synthetic data and variance files for use by the inference workflow.

**Outputs:**

- ``p_true.txt`` — True parameter vector used for data generation.
- ``ydata.txt`` — Synthetic observational data of shape
  ``(outdim, neach)``.
- ``ydatavar.txt`` — Data-noise variance array of shape ``(outdim,)``.

**Example:**

.. code-block:: bash

   python create_data_truepc.py -s 0.5 -e 7
   python create_data_truepc.py -s 0.1 -e 3 -c p_true_input.txt

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - ``-s, --sig``
     - ``0.5``
     - Noise standard deviation added to the PC model output.
   * - ``-e, --each``
     - ``1``
     - Number of replicate samples per output location.
   * - ``-c, --cfile``
     - ``None``
     - True-coefficients file.  If omitted, random values are drawn.


prep_data.x
-----------

An alternative Bash script for manually preparing ``ydata.txt`` and
``ydatavar.txt``.  Users can edit this script to hard-code their own
observations and noise variances.

**Example:**

.. code-block:: bash

   bash prep_data.x


run_infer.py
------------

Perform MCMC-based Bayesian inference using the PC surrogate.

This script loads the PC surrogate from ``results.pk``, configures the
likelihood, prior, and adaptive MCMC sampler, and runs Bayesian
calibration.  Parameters with total sensitivity below 1 % are
automatically fixed at their nominal values.

**Expected inputs:**

- ``results.pk`` — PC surrogate from the UQPC workflow.
- ``ydata.txt`` — Observational data.
- ``ydatavar.txt`` — Data-noise variance.

**Outputs:**

- ``calib_results.pk`` — Pickled calibration dictionary containing the
  MCMC chain, MAP parameters, posterior object, and diagnostics.
- ``chain.txt`` — MCMC chain (plain text).
- ``mapparams.txt`` — MAP parameter estimate.
- ``surr_error_var.txt`` — Surrogate approximation error variance.

**Example:**

.. code-block:: bash

   # Classical inference (no model error)
   python run_infer.py

   # Inference with embedded model error
   python run_infer.py -m

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - ``-m, --merr``
     - ``False``
     - Enable embedded model-error inference.  When set, a Gaussian
       marginal likelihood (``gausmarg``) is used with additional
       stochastic output-order terms.

.. note::

   Key inference settings (MCMC length, burn-in, adaptation schedule,
   prior type, likelihood type, etc.) are configured inside the script.
   Users should edit the ``model_infer_config`` dictionary to match
   their problem.


postp_infer.py
--------------

Post-process and visualise Bayesian inference results.

This script loads ``calib_results.pk`` and ``results.pk``, then
generates:

- **Chain diagnostics** — MCMC trace plots for each inferred dimension
  (``chain.png``).
- **Posterior predictive** — Model predictions from posterior samples
  evaluated through the surrogate.
- **Input posterior PDFs** — Triangular pair-plot of parameter posteriors
  (``pdf_tri_post_inputs.png``).
- **Output posterior PDFs** — Triangular pair-plot of output posteriors
  with observed data overlaid (``pdf_tri_post_outputs.png``).
- **Prior vs posterior** — Kernel-density and scatter comparisons between
  training-set (prior) and posterior parameter samples
  (``pdf_prior_vs_post.png``).
- **Posterior predictive envelopes** — Output predictions at MAP, with
  data and error bars (``postpred_map.png``).

**Expected inputs:**

- ``results.pk`` — PC surrogate results.
- ``calib_results.pk`` — Calibration results from ``run_infer.py``.
- ``ydata.txt`` — Observational data.
- ``ydatavar.txt`` — Data-noise variance.
- ``surr_error_var.txt`` — Surrogate error variance.

**Example:**

.. code-block:: bash

   # Without known true parameters
   python postp_infer.py

   # With known true parameters overlaid on posteriors
   python postp_infer.py -c p_true.txt

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - ``-c, --cfile``
     - ``None``
     - True-parameters file.  When provided, the true values are
       overlaid on posterior PDF plots for comparison.

.. note::

   Parameter and output names are read from ``pnames.txt`` and
   ``outnames.txt`` when present; otherwise generic names (``Par_``,
   ``Out_``) are used.


workflow_iuq.x
--------------

An annotated Bash script that demonstrates the complete inverse-UQ
workflow: data preparation, inference (with or without model error),
and post-processing.

Run it from the ``apps/iuq/`` directory (after having run the
UQPC forward workflow to produce ``results.pk``):

.. code-block:: bash

   bash workflow_iuq.x
