#!/usr/bin/env python

"""This file is for testing the NN wrapper class with scalar valued functions.
    Calling the NN constructor passes in an example net_options dictionary, while the build and evaluate
    sections pass in both dictionaries and explicit keyword arguments during their respective function calls."""

import sys
import torch
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

    # (1) Initialize neural network with optional parameters for object instantiation
    net_options = {'wp_function': Poly(0),
                   'indim': ndim, 
                   'outdim': nout,
                   'layer_pre': True,
                   'layer_post': True,
                   'biasorno': True,
                   'nonlin': True,
                   'mlp': False, 
                   'final_layer': None,
                   'device': device,
                   }

    # Pass in unpacked net_options dict to constructor through kwargs
    nnet = NN('RNet', 3, 3, **net_options)
    
    # (1.5) Split data into training and validation, assign through member functions
    ntrn = int(trn_factor * nall)
    indperm = range(nall) # np.random.permutation(range(nall))
    indtrn = indperm[:ntrn]
    indval = indperm[ntrn:]
    xtrn, xval = xall[indtrn, :], xall[indval, :]
    ytrn, yval = yall[indtrn, :], yall[indval, :]

    nnet.set_validation_data(xval, yval) # optional
    nnet.set_training_data(xtrn, ytrn)

    # (2) 1st build: Call build function (defaults to UQ method of variational inference) with optional parameters for fitting
    nnet.build(datanoise=datanoise, lrate=0.01, batch_size=None, nsam=1, nepochs=5000, verbose=False)

    results = nnet.evaluate(xtst, nsam = 100, msc = 2) # Return samples of predictions with variance + covariance
    print("Y_eval:", results['Y_eval'], end="\n\n") # Printing only samples of predictions

    # (3) 2nd build: Example of throwing an exception when passing in a network option through build()
    # fit_options = {'datanoise': 0.05,
    #                'outdim': 3,
    #                'lrate': 0.01,
    #                'batch_size': None,
    #                'nsam': 1,
    #                'nepochs': 300,
    #                'verbose': False
    #                }

    # nnet.build(**fit_options)
    # results = nnet.evaluate(xtst, nsam = 100, msc = 2)
    # print(results)


if __name__ == '__main__':
    main()