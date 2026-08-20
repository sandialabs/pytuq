.. pytuq documentation master file, created by
   sphinx-quickstart on Mon Aug  5 13:27:58 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


PyTUQ Documentation
============================================


.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - |python-tests| |deployment| |coveralls|

.. |python-tests| image:: https://github.com/sandialabs/pytuq/actions/workflows/python-test.yml/badge.svg
    :alt: Python Tests Build Status
    :target: https://github.com/sandialabs/pytuq/actions/workflows/python-test.yml

.. |deployment| image:: https://github.com/sandialabs/pytuq/actions/workflows/documentation.yml/badge.svg
    :alt: Deployment Status
    :target: https://github.com/sandialabs/pytuq/actions/workflows/documentation.yml

.. |coveralls| image:: https://coveralls.io/repos/github/sandialabs/pytuq/badge.svg?branch=main
    :target: https://coveralls.io/github/sandialabs/pytuq?branch=main
    :alt: Coverage Status

.. end-badges


**Last Updated**: |today| 

**Version**: |version|

Hello, `PyTUQ <https://github.com/sandialabs/pytuq>`__ is a Python-only toolkit for uncertainty quantification in computational models.
To explore the modules offered by PyTUQ, use the navigation panel on the left or below.

Check out the :ref:`Getting Started <about>` section for further information, including how to
:ref:`install <installation>` the project.

.. toctree::
   :maxdepth: 4
   :caption: Getting Started

   misc/installation
   misc/about

.. toctree::
   :maxdepth: 4
   :caption: Examples

   examples/pc
   examples/regression
   examples/inference
   examples/sensitivity
   examples/rosenblatt
   examples/rv
   examples/functions
   examples/plotting

.. toctree::
   :maxdepth: 4
   :caption: Apps

   apps/pc
   apps/surrogates
   apps/plot
   apps/uqpc
   apps/iuq
   apps/other
   apps/awkies

.. toctree::
   :maxdepth: 4
   :caption: Tutorials

   auto_examples/index

.. toctree::
   :maxdepth: 4
   :caption: List of Modules

   autoapi/pytuq/fit/index
   autoapi/pytuq/ftools/index
   autoapi/pytuq/func/index
   autoapi/pytuq/gsa/index
   autoapi/pytuq/linred/index
   autoapi/pytuq/lreg/index
   autoapi/pytuq/minf/index
   autoapi/pytuq/optim/index
   autoapi/pytuq/rv/index
   autoapi/pytuq/surrogates/index
   autoapi/pytuq/utils/index
   autoapi/pytuq/workflows/index

.. toctree::
   :maxdepth: 4
   :caption: Misc

   misc/indices
   misc/references
   misc/diagram