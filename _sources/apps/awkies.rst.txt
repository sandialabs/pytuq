================
Shell Utilities
================

The ``apps/awkies/`` directory contains lightweight Bash/AWK scripts for
quick manipulation of text-based data files.  They operate on
whitespace-delimited matrices and are useful for pre- and post-processing
data outside of Python.


getrange.x
----------

Compute per-dimension ranges of a sample matrix.

Given an :math:`N \times d` data file, this script prints a
:math:`d \times 2` range table (min, max per column).  An optional
cushion fraction expands each range by a specified fraction of its width.

**Usage:**

.. code-block:: bash

   getrange.x samples.dat              # exact ranges
   getrange.x samples.dat 0.05         # 5 % cushion on each side

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Argument
     - Description
   * - ``<filename>``
     - Whitespace-delimited data file (:math:`N \times d`).
   * - ``[cushion_fraction]``
     - Optional fraction (default 0) to pad each range symmetrically.


scale.x
-------

Scale matrix data between a given parameter domain and :math:`[-1, 1]^d`.

Reads an :math:`N \times d` data file and a :math:`d \times 2` domain
file (one ``min max`` row per dimension), then scales each column *to*
or *from* the unit hypercube.

**Usage:**

.. code-block:: bash

   scale.x input.dat to   domain.dat output.dat   # [-1,1] -> domain
   scale.x input.dat from domain.dat output.dat   # domain -> [-1,1]

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Argument
     - Description
   * - ``<input>``
     - Input data file.
   * - ``<to|from>``
     - Direction: ``to`` maps from :math:`[-1,1]` to the domain;
       ``from`` maps from the domain to :math:`[-1,1]`.
   * - ``<domain>``
     - Domain file (:math:`d` rows, 2 columns: min and max).
   * - ``<output>``
     - Output file for the scaled data.


transpose.x
------------

Transpose a whitespace-delimited matrix file.

Reads an :math:`N \times d` file and writes a :math:`d \times N` file
to standard output.

**Usage:**

.. code-block:: bash

   transpose.x matrix.dat > matrix_T.dat
