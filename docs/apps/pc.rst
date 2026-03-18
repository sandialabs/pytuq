========================
Polynomial Chaos 
========================

These scripts in ``apps/pc/`` provide standalone command-line tools for
building PC surrogates, preparing input PC representations, and sampling
from PC random variables.  They can be used independently or as building
blocks within the :doc:`Forward UQ <uqpc>` workflow.


pc_fit.py
---------

Fit a Polynomial Chaos expansion to input/output data.

Given training inputs and outputs, this script fits a PC surrogate using
analytical projection (``anl``), least-squares (``lsq``), or Bayesian
Compressed Sensing (``bcs``).  A fraction of the data is held out for
testing.  It produces parity plots, per-sample fit comparisons, a
sensitivity bar chart, and saves the surrogate objects.

**Outputs:**

- ``pcrv.pk`` — Pickled ``PCRV`` surrogate object.
- ``lregs.pk`` — Pickled linear regression objects.
- ``dm_*.png`` — Parity (model vs approximation) plots per output.
- ``fit_s*.png`` — Per-sample overlay of original and PC-predicted
  outputs.
- ``sens_pc.png`` — Global sensitivity bar chart.

**Examples:**

.. code-block:: bash

   python pc_fit.py -x ptrain.txt -y ytrain.txt -m bcs -c LU -o 3
   python pc_fit.py -x ptrain.txt -y ytrain.txt -m lsq -c HG -o 2 -t 0.8

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - ``-x, --xdata``
     - ``ptrain.txt``
     - Input data file.
   * - ``-y, --ydata``
     - ``ytrain.txt``
     - Output data file.
   * - ``-d, --xcond``
     - ``None``
     - Conditioning x-grid file (e.g. time or spatial grid).  If
       omitted, an integer index is used.
   * - ``-q, --outnames_file``
     - ``outnames.txt``
     - Output names file.
   * - ``-p, --pnames_file``
     - ``pnames.txt``
     - Parameter names file.
   * - ``-t, --trnfactor``
     - ``0.9``
     - Fraction of data used for training (remainder is held out for
       testing).
   * - ``-m, --method``
     - ``bcs``
     - Fitting method: ``lsq``, ``bcs``, or ``anl``.
   * - ``-c, --pctype``
     - ``LU``
     - PC basis type: ``LU`` or ``HG``.
   * - ``-o, --order``
     - ``1``
     - Polynomial chaos order.

.. note::

   Parameter and output names are read from ``pnames.txt`` and
   ``outnames.txt`` when present; otherwise generic names are used.


pc_prep.py
----------

Generate a PC coefficient file encoding the input random variable.

This script accepts one of three input formats and produces
``pcf.txt`` — a PC coefficient matrix with one column per input
dimension:

* ``marg`` — Per-dimension marginal PC coefficients (one line per
  dimension: mean, first-order coeff, ...).
* ``sam`` — Raw samples; a Rosenblatt/PC transform is fitted.
* ``mvn`` — Mean vector file + covariance matrix file for a
  multivariate normal (Cholesky decomposition is used).

**Outputs:** ``pcf.txt``

**Examples:**

.. code-block:: bash

   # From marginal PCs (e.g. mean and half-width per dimension)
   python pc_prep.py -f marg -i param_margpc.txt -p 1

   # From raw samples
   python pc_prep.py -f sam -i xsam.txt -p 2 -t HG

   # From multivariate normal (mean + covariance)
   python pc_prep.py -f mvn -i mean.txt -c cov.txt

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - ``-f, --fmt``
     - ``sam``
     - Input format: ``marg``, ``sam``, or ``mvn``.
   * - ``-i, --inp``
     - ``xsam.txt``
     - Input filename.  For ``marg``: marginal coefficient file; for
       ``sam``: samples file; for ``mvn``: mean vector file.
   * - ``-c, --cov``
     - ``cov.txt``
     - Covariance filename (only used when format is ``mvn``).
   * - ``-p, --pco``
     - ``1``
     - PC order (used for ``marg`` and ``sam`` formats).
   * - ``-t, --pct``
     - ``HG``
     - PC type: ``LU`` or ``HG`` (used for ``sam`` format).


pc_sam.py
---------

Sample from a multivariate PC random variable.

Given a PC coefficient file (e.g. ``pcf.txt`` from ``pc_prep.py``),
this script constructs a ``PCRV`` object, draws germ samples, evaluates
the PC expansion, and writes both germ and physical-space samples.

**Outputs:**

- ``psam.txt`` — Samples in physical (PC-transformed) space.
- ``qsam.txt`` — Samples in germ (standard normal or uniform) space.

**Example:**

.. code-block:: bash

   python pc_sam.py -f pcf.txt -t HG -n 500

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - ``-f, --pcf``
     - ``pcf.txt``
     - PC coefficient file.  Each column is the PC coefficient vector
       for one input dimension.
   * - ``-t, --pct``
     - ``HG``
     - PC type: ``LU`` or ``HG``.
   * - ``-n, --nsam``
     - ``111``
     - Number of samples to draw.
