# Python Toolkit for Uncertainty Quantification (PyTUQ)

#### Khachik Sargsyan, Bert Debusschere, Emilie Grace Baillo


[![Deploy to GitHub Pages](https://github.com/sandialabs/pytuq/actions/workflows/documentation.yml/badge.svg)](https://github.com/sandialabs/pytuq/actions/workflows/documentation.yml)
[![Run Tests](https://github.com/sandialabs/pytuq/actions/workflows/python-test.yml/badge.svg)](https://github.com/sandialabs/pytuq/actions/workflows/python-test.yml)
[![Coverage Status](https://coveralls.io/repos/github/sandialabs/pytuq/badge.svg?branch=main)](https://coveralls.io/github/sandialabs/pytuq?branch=main)



## Overview

The Python Toolkit for Uncertainty Quantification (PyTUQ) is a Python-only collection of libraries and tools designed for quantifying uncertainty in computational models. PyTUQ offers a range of UQ functionalities, including Bayesian inference and linear regression methods, polynomial chaos expansions, and global sensitivity analysis methods. PyTUQ features advanced techniques for dimensionality reduction, such as SVD and Karhunen-Loeve expansions, along with various MCMC methods for calibration and inference. The toolkit also includes robust classes for multivariate random variables and integration techniques, making it a versatile resource for researchers and practitioners seeking to quantify uncertainty in their numerical predictions. To explore the PyTUQ documentation and learn more, visit our website [here](https://sandialabs.github.io/pytuq/).

## Dependencies
PyTUQ requires:
* numpy
* scipy
* matplotlib

Optional dependencies include:
* pytorch (NN surrogates)
* QUiNN (Quantification of Uncertainties in Neural Networks)
* pyswarms (Particle Swarm Optimization)
* dill (for saving python objects)

## Installation

### Basic Installation
Install PyTUQ with core dependencies:
```bash
$ pip install pytuq
```

#### Optional Features

Install additional features as needed:
* Neural network surrogates
```bash
$ pip install 'pytuq[nn]'
```

* Particle swarm optimization
```bash
$ pip install 'pytuq[optim]'
```

* All optional features
```bash
$ pip install 'pytuq[all]'
```

* Development tools
```bash
$ pip install 'pytuq[dev]'
```

### Installation from Source
1. To install PyTUQ from source, start up a Python virtual environment and clone the repository:
```bash
    $ source <PYTHON_VENV_DIR>/bin/activate
    $ git clone git@github.com:sandialabs/pytuq.git
    $ cd pytuq
```
2. (Optional) To take advantage of the neural network surrogate model capabilites in PyTUQ, use the following command to install PyTUQ's optional dependencies:
```
    $ pip install -r requirements.txt
```
3. Install primary PyTUQ dependencies and PyTUQ:
```
    $ pip install .
```

## Contributors
Habib N. Najm (Sandia National Laboratories)
Javier Murgoitio-Esandi (Google)
Cosmin Safta (Sandia National Laboratories)
Joy Bahr-Mueller (Sandia National Laboratories)
Vahan Sargsyan (Stuyvesant High School)

## License
Distributed under BSD 3-Clause License. See `LICENSE.txt` for more information.

## Acknowledgements
This work is supported by the Scientific Discovery through Advanced Computing (SciDAC) Program under the Office of Science at the U.S. Department of Energy. 

Sandia National Laboratories is a multimission laboratory managed and operated by National Technology & Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell International Inc., for the U.S. Department of Energy’s National Nuclear Security Administration under contract DE-NA0003525.
