====================
Multioutput Fits
====================

These scripts fit surrogate models to multioutput data.  Each takes an
input/output dataset, splits it into training and testing subsets, fits
the surrogate, and produces parity plots, per-sample overlays, and
(where applicable) sensitivity bar charts.


kl_fit.py
---------

Build Karhunen–Loève (KL) decompositions of multioutput data.

This script reads a multioutput dataset, performs a KL expansion to
reduce its dimensionality, and generates diagnostic plots including
explained-variance curves, data-vs-approximation scatter plots, and
sample-wise fit comparisons.

**Outputs:**
  - ``dm_*.png`` — Diagonal model-vs-approximation scatter plots per output.
  - ``fit_s*.png`` — Per-sample overlay of original and KL-reconstructed outputs.

**Example:**

.. code-block:: bash

   python kl_fit.py -y ydata.txt -e 5

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - ``-d, --xcond``
     -
     - Conditioning x-grid file (default: index array).
   * - ``-y, --ydata``
     - ``ydata.txt``
     - Multioutput data file.
   * - ``-e, --neig``
     -
     - Number of eigenvalues to retain (default: auto at 99%).


klsurr_fit.py
-------------

Build KL-based reduced-dimensional surrogates of multioutput models.

This script combines a Karhunen–Loève expansion with surrogate modelling
(Polynomial Chaos or Neural Network) to construct a reduced-dimensional
surrogate for multioutput data.  It produces parity plots, per-sample fit
comparisons, and global sensitivity bar charts.

**Outputs:**
  - ``dm_*.png`` — Parity plots per output.
  - ``fit_s*.png`` — Per-sample overlay of model and KL+surrogate predictions.
  - ``sens_klsurr.png`` — Global sensitivity bar chart.
  - ``klsurr.pk`` — Pickled ``KLSurr`` object.

**Example:**

.. code-block:: bash

   python klsurr_fit.py -x ptrain.txt -y ytrain.txt -s PC -m bcs -o 3

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
     -
     - Conditioning x-grid file.
   * - ``-q, --outnames_file``
     - ``outnames.txt``
     - Output names file.
   * - ``-p, --pnames_file``
     - ``pnames.txt``
     - Parameter names file.
   * - ``-t, --trnfactor``
     - 0.9
     - Fraction of data used for training.
   * - ``-s, --surr``
     - ``PC``
     - Surrogate type: ``PC`` or ``NN``.
   * - ``-m, --method``
     - ``bcs``
     - Fitting method: ``lsq``, ``bcs``, or ``anl``.
   * - ``-c, --pctype``
     - ``LU``
     - PC basis type: ``LU`` or ``HG``.
   * - ``-o, --order``
     - 1
     - Polynomial chaos order.


nn_fit.py
---------

Build neural-network-based surrogates for multioutput models.

This script trains a Residual Network (``RNet`` from QUiNN) on
user-supplied input/output data, splits the data into training and
testing sets, and produces parity plots and per-sample fit comparisons.

Requires the `QUiNN <https://github.com/sandialabs/quinn>`_ package.

**Outputs:**
  - ``dm_*.png`` — Parity (model vs approximation) plots per output.
  - ``fit_s*.png`` — Per-sample overlay of original and NN-predicted outputs.

**Example:**

.. code-block:: bash

   python nn_fit.py -x ptrain.txt -y ytrain.txt -t 0.8

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
     -
     - Conditioning x-grid file.
   * - ``-q, --outnames_file``
     - ``outnames.txt``
     - Output names file.
   * - ``-p, --pnames_file``
     - ``pnames.txt``
     - Parameter names file.
   * - ``-t, --trnfactor``
     - 0.9
     - Fraction of data used for training.
