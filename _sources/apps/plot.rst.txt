=============
Plotting Apps
=============

These scripts live in ``apps/plot/`` and provide quick command-line
visualisations for common UQ data types.


plot_cov.py
-----------

Plot 2-D marginal covariance ellipses for a multivariate normal.

This script reads a mean vector and covariance matrix and produces
pairwise 2-D covariance ellipse plots as well as a triangular grid of
all pairs.

**Example:**

.. code-block:: bash

   python plot_cov.py -m mean.txt -c cov.txt 0 1 2

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - positional
     - all
     - Indices of parameters to show.
   * - ``-m, --mean``
     - ``mean.txt``
     - Mean file.
   * - ``-c, --cov``
     - ``cov.txt``
     - Covariance file.


plot_ens.py
-----------

Plot an ensemble of output-data curves (spaghetti plot).

**Outputs:** ``ensemble.png``

**Example:**

.. code-block:: bash

   python plot_ens.py -y ytrain.dat

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - ``-y, --ydata``
     - ``ytrain.dat``
     - Output data file.


plot_pcoord.py
--------------

Plot parallel-coordinate diagrams for multivariate data.

Data are normalised to [-1, 1] before plotting.  Optional label files
allow colour-coding by group.

**Outputs:** ``pcoord_*.png``

**Example:**

.. code-block:: bash

   python plot_pcoord.py -x ptrain.txt -y ytrain.txt -e 5

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
     -
     - Optional output data file.
   * - ``-o, --outnames_file``
     - ``outnames.txt``
     - Output names file.
   * - ``-p, --pnames_file``
     - ``pnames.txt``
     - Parameter names file.
   * - ``-e, --every``
     - 1
     - Sample thinning factor.
   * - ``-l, --labels_file``
     -
     - Label file for group colouring.
   * - ``-c, --ndcut``
     - 0
     - Chunk size for splitting dimensions (0 = all).


plot_pdfs.py
------------

Plot probability density functions from MCMC or other samples.

Supports triangular pair-plots, individual marginal PDFs (histograms or
KDEs), burn-in trimming, thinning, prior-range overlays, and nominal-value
markers.

**Example:**

.. code-block:: bash

   python plot_pdfs.py -p pchain.dat -t tri -b 1000 -e 5

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - positional
     - all
     - Indices of parameters to show.
   * - ``-p, --samples_file``
     - ``pchain.dat``
     - Samples file.
   * - ``-n, --names_file``
     -
     - Parameter names file.
   * - ``-l, --nominal_file``
     -
     - Nominal parameter values file.
   * - ``-g, --prange_file``
     -
     - Prior range file.
   * - ``-t, --plot_type``
     - ``tri``
     - ``tri``, ``ind``, or ``inds``.
   * - ``-f, --pdf_type``
     - ``hist``
     - ``hist`` or ``kde``.
   * - ``-b, --burnin``
     - 0
     - Burn-in samples to discard.
   * - ``-e, --every``
     - 1
     - Thinning interval.


plot_xx.py
----------

Plot pairwise scatter plots of input data, optionally colour-coded by
label.

**Outputs:** ``xx_<dim1>_<dim2>.png``

**Example:**

.. code-block:: bash

   python plot_xx.py -x qtrain.txt -e 2

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - ``-x, --xdata``
     - ``qtrain.txt``
     - Input data file.
   * - ``-p, --pnames_file``
     - ``pnames.txt``
     - Parameter names file.
   * - ``-e, --every``
     - 1
     - Sample thinning factor.
   * - ``-l, --labels_file``
     -
     - Label file for group colouring.


plot_yx.py
----------

Plot outputs versus one input dimension at a time (linear and log scale).

**Outputs:** ``yx_<outname>.png``, ``yx_<outname>_log.png``

**Example:**

.. code-block:: bash

   python plot_yx.py -x qtrain.txt -y ytrain.txt -c 4 -r 3

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - ``-x, --xdata``
     - ``qtrain.txt``
     - Input data file.
   * - ``-y, --ydata``
     - ``ytrain.txt``
     - Output data file.
   * - ``-o, --outnames_file``
     - ``outnames.txt``
     - Output names file.
   * - ``-p, --pnames_file``
     - ``pnames.txt``
     - Parameter names file.
   * - ``-e, --every``
     - 1
     - Sample thinning factor.
   * - ``-c, --cols``
     - 4
     - Number of subplot columns.
   * - ``-r, --rows``
     - 6
     - Number of subplot rows.


plot_yxx.py
-----------

Plot outputs versus pairs of inputs in a triangular layout, colour-coded
by the output value.  Useful for identifying interaction effects.

**Outputs:** ``yxx_<iout>.png``

**Example:**

.. code-block:: bash

   python plot_yxx.py -x qtrain.txt -y ytrain.txt -e 2

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - ``-x, --xdata``
     - ``qtrain.txt``
     - Input data file.
   * - ``-y, --ydata``
     - ``ytrain.txt``
     - Output data file.
   * - ``-o, --outnames_file``
     - ``outnames.txt``
     - Output names file.
   * - ``-p, --pnames_file``
     - ``pnames.txt``
     - Parameter names file.
   * - ``-e, --every``
     - 1
     - Sample thinning factor.
