==========
Other Apps
==========


create_data.py
--------------

Generate training data for benchmark functions.

This script evaluates a chosen benchmark function at random or user-supplied
input points and saves the resulting input/output pairs to text files.
Optionally, it can also compute and save gradients.

Functions are discovered dynamically from PyTUQ's benchmark sub-modules
(``bench``, ``bench1d``, ``bench2d``, ``benchNd``, ``chem``, ``genz``,
``poly``, ``toy``).  Run ``python create_data.py -h`` to see the full
list of available functions.

**Outputs:**

- ``xtrain.txt`` — Input sample array of shape ``(n, d)``.
- ``ytrain.txt`` — Output array of shape ``(n, m)``.
- ``gtrain.txt`` — *(only with* ``-g`` *)* Gradient array of shape
  ``(n, m*d)``.

**Examples:**

.. code-block:: bash

   python create_data.py -f Ishigami -n 200 -s 0.01
   python create_data.py -f Genz1 -n 500 -g -z 42
   python create_data.py -f Muller-Brown -x mysamples.txt

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Default
     - Description
   * - ``-n, --npts``
     - ``100``
     - Number of sample points.  Ignored when ``-x`` is given.
   * - ``-f, --func``
     - ``Muller-Brown``
     - Benchmark function name (see ``--help`` for choices).
   * - ``-x, --xtrain``
     - ``None``
     - Optional file of pre-generated input samples.  When provided,
       the number of samples is determined by the file.
   * - ``-s, --sigma``
     - ``0.0``
     - Standard deviation of additive Gaussian noise on outputs.
   * - ``-g``
     - off
     - Compute and save gradients.
   * - ``-z, --seed``
     - ``None``
     - Random seed for reproducibility.


mrun.py
-------

Utilities for parallel execution of shell tasks via multiprocessing.

This module provides helper functions for running batches of shell
commands in parallel, each in its own working directory.  Two strategies
are offered:

- ``mrun`` — spawns one ``multiprocessing.Process`` per task.
- ``mpool`` — uses a ``multiprocessing.Pool`` with a fixed worker count.

When executed as a script it reads a ``tasks`` file (each line:
``<directory> <command>``) and dispatches all tasks with ``mpool``.

**Input file:** ``tasks`` — one task per line in the format
``<directory> <command>``.

**Example:**

.. code-block:: bash

   python mrun.py 4          # run tasks from ./tasks using 4 workers

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Argument
     - Description
   * - positional (``nproc``)
     - Number of worker processes.
