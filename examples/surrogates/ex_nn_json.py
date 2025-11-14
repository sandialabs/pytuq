r"""
Residual Neural Network Construction (JSON)
=============================================

This file is for testing the NN wrapper class with scalar valued functions.
Calls the NN constructor with no optional keyword arguments before passing in an example json file with user-defined
fitting and building options. The build and evaluate sections pass in explicit keyword arguments 
during their respective function calls to demonstrate updates to the neural network fitting/evaluating options.
When requested with a provided filename, the updated options are printed out to a json file.
"""

import sys
import torch
import json
import numpy as np

from pytuq.surrogates.nn import NN
from quinn.solvers.nn_vi import NN_VI
from quinn.nns.rnet import RNet, Poly
from quinn.utils.plotting import myrc
from quinn.utils.maps import scale01ToDom
from quinn.func.funcs import Sine, Sine10, blundell

def main():
    """Main function."""
    torch.set_default_dtype(torch.double)
    myrc()

    #################################################################################
    #################################################################################

    # defaults to cuda:0 if available
    device_id='cuda:0'
    device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
    print("Using device",device)

    nall = 15 # total number of points
    trn_factor = 0.9 # which fraction of nall goes to training
    ntst = 13 # separate test set
    ndim = 1 # input dimensionality
    datanoise = 0.02 # Noise in the generated data
    true_model, nout = Sine, 1  # Scalar valued output example

    #################################################################################
    #################################################################################

    # Domain: defining range of input variable
    domain = np.tile(np.array([-np.pi, np.pi]), (ndim, 1))

    np.random.seed(111)

    # Generating x-y training, validation, and testing data
    xall = scale01ToDom(np.random.rand(nall, ndim), domain)
    if true_model is not None:
        yall = true_model(xall, datanoise=datanoise)

    if ntst > 0:
        np.random.seed(100)
        xtst = scale01ToDom(np.random.rand(ntst, ndim), domain)
        if true_model is not None:
            ytst = true_model(xtst, datanoise=datanoise)

    # (1) Initialize neural network: call NN constructor without kwargs, read in json file options
    nnet = NN('RNet', 3, 3)

    nnet.read_options_from_file('options.json')

    # (1.5) Split data into training and validation, assign through member functions
    ntrn = int(trn_factor * nall)
    indperm = range(nall) # np.random.permutation(range(nall))
    indtrn = indperm[:ntrn]
    indval = indperm[ntrn:]
    xtrn, xval = xall[indtrn, :], xall[indval, :]
    ytrn, yval = yall[indtrn, :], yall[indval, :]

    nnet.set_validation_data(xval, yval) # optional
    nnet.set_training_data(xtrn, ytrn)

    # (2) Call build function (defaults to UQ method of variational inference) with optional parameters for neural network 
    nnet.build(nepochs=5000)

    # Pass in optional filename to write NN options to file
    nnet.get_options(json_print = 'current_options.json')

    results = nnet.evaluate(xtst, nsam = 100, msc = 2) # Return samples of predictions with variance + covariance
    print("Y_eval:", results['Y_eval'], end="\n\n") # Printing only samples of predictions

    options = nnet.get_options() # Printing out dictionary of final options
    print(options)


if __name__ == '__main__':
    main()