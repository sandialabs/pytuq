.. highlight:: shell

============
Installation
============

.. _installation:

Install from source
--------------------

The sources for PyTUQ can be downloaded from the `PyTUQ GitHub repo`_.

To install from source for the most recent stable release of PyTUQ, start up a Python virtual environment and clone the GitHub repository:

.. code-block:: console

    $ source <PYTHON_VENV_DIR>/bin/activate # starting up an example Python virtual environment
    $ git clone git@github.com:sandialabs/pytuq.git
    $ cd pytuq

Then, install the PyTUQ package and its dependencies via 'pip'.

To install just the primary PyTUQ dependencies, use the following command:

.. code-block:: console

    $ pip install .

To take advantage of the neural network surrogate model capabilities in PyTUQ, use the following commands to install PyTUQ along with its optional dependencies:

.. code-block:: console

    $ pip install -r requirements.txt
    $ pip install .

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.





.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _PyTUQ GitHub repo: https://github.com/sandialabs/pytuq
