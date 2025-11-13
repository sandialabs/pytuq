.. highlight:: shell

============
Installation
============

.. _installation:

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

Basic Installation
------------------
 
PyTUQ can be installed directly from PyPI using pip:
 
.. code-block:: console
 
    $ pip install pytuq
 
This will install PyTUQ with its core dependencies (numpy, scipy, matplotlib).
 
Optional Features
-----------------
 
PyTUQ provides several optional feature sets that can be installed as needed:
 
**Neural Network Surrogates**
 
To use neural network surrogate models (requires PyTorch and QUiNN):
 
.. code-block:: console
 
    $ pip install 'pytuq[nn]'
 
**Particle Swarm Optimization**
 
To use particle swarm optimization capabilities (requires pyswarms):
 
.. code-block:: console
 
    $ pip install 'pytuq[optim]'
 
**All Optional Features**
 
To install all optional features at once:
 
.. code-block:: console
 
    $ pip install 'pytuq[all]'
 
**Development Tools**
 
To install development dependencies (useful for contributors):
 
.. code-block:: console
 
    $ pip install 'pytuq[dev]'
 
**Documentation Tools**

To build the documentation, you will need to install the documentation dependencies:

.. code-block:: console

    $ pip install 'pytuq[doc]'
 
Install from Source
-------------------
 
The sources for PyTUQ can be downloaded from the `PyTUQ GitHub repo`_.
 
To install from source for the most recent stable release of PyTUQ, start up a Python virtual environment and clone the GitHub repository:
 
.. code-block:: console
 
    $ source <PYTHON_VENV_DIR>/bin/activate
    $ git clone git@github.com:sandialabs/pytuq.git
    $ cd pytuq
 
Then, install the PyTUQ package and its dependencies via pip. You can install just the core dependencies:
 
.. code-block:: console
 
    $ pip install .
 
Or install with optional features using the extras syntax shown above:
 
.. code-block:: console
 
    $ pip install '.[nn]'      # for neural network features
    $ pip install '.[all]'     # for all optional features
    $ pip install '.[dev]'     # for development tools



.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _PyTUQ GitHub repo: https://github.com/sandialabs/pytuq
