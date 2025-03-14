# Python Toolkit for Uncertainty Quantification (PyTUQ)

#### Khachik Sargsyan, Bert Debusschere, Emilie Grace Baillo

## Overview

Python Toolkfit for Uncertainty Quantification (PyTUQ) is a Python-only set of tools for uncertainty quantification.

## Dependencies
PyTUQ requires:
* numpy
* scipy
* matplotlib
* pytorch
* QUINN (optional)

## Installation
1. To install PyTUQ from source, start up a Python virtual environment and clone the repository:
```
    $ source <PYTHON_VENV_DIR>/bin/activate
    $ git clone git@github.com:sandialabs/pytuq.git
    $ cd pytuq
```
2. Install PyTUQ dependencies and PyTUQ via `pip`:
```
    $ pip install -r requirements.txt # installs QUINN
    $ pip install .  
```

## License
Distributed under BSD 3-Clause License. See `LICENSE.txt` for more information.

## Acknowledgements
This work is supported by the Scientific Discovery through Advanced Computing (SciDAC) Program under the Office of Science at the U.S. Department of Energy. 

Sandia National Laboratories is a multimission laboratory managed and operated by National Technology & Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell International Inc., for the U.S. Department of Energy’s National Nuclear Security Administration under contract DE-NA0003525.
