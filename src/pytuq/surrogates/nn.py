#!/usr/bin/env python

"""This module provides a QUiNN (Quantification of Uncertainties in Neural Networks) wrapper class to facilitate 
    the universal coupling of FASTMath UQ tools and libraries. This class focuses on the use case 
    of a Residual Neural Network fitted with variational inference, keeping in mind 
    flexibility to implement additional UQ functionalities in the future.

    The NN class supports a minimal API, with methods to construct the model, build it with training data,
    evaluate it with input data, and offer samples of predictions with variance and covariance.

    Note:
        The current implementation focuses on providing a general foundation for Neural Network surrogate model creation. 
        While not all construct and build options are currently supported, the class was developed with future growth in mind.
"""

import torch
import json
import numpy as np

from quinn.nns.rnet import RNet, LayerFcn, Poly, Const, Lin, Quad, Cubic, NonPar
from quinn.solvers.nn_vi import NN_VI


class NN:
    """A wrapper class to access QUiNN functionalities for Neural Network surrogate models. 

    Attributes:
        nnet (RNet object): Neural network, defaults to Residual Neural Network object.
        type (str): Type of neural network, defaults to 'RNet'.
        n_layers (int): Layers in neural network.
        n_nodes (int): Nodes per layer.
        uqnet (NN_VI object): UQ method, defaults to variational inference object and method.
        options (dict): Dictionary with user-specified options for instantiation of Neural Network, building, and evaluating.
        _x_trn (np.ndarray): Input training data.
        _y_trn (np.ndarray): Output training data.
        _x_val (np.ndarray, optional): Input validation data.
        _y_val (np.ndarray, optional): Output validation data. 
    """

    def __init__(self, type, n_layers, n_nodes, **kwargs):
        """Initializes a Residual Neural Network (NNet) object with, at minimum:
        numbers of layers, number of nodes, and neural network type (defaulting to RNet).

        Args:
            n_layers (int): Number of layers.
            n_nodes (int): Width of the RNet, i.e. number of units in each hidden layer.            
        """

        # Input validation for type, n_layers, n_nodes
        if type != "RNet":
            raise ValueError("Type of NN must be a Residual Neural Network, RNet.")
        else:
            self.type = type
        
        if n_layers <= 0 or n_nodes <= 0:
            raise ValueError("n_layers and n_nodes must be positive integers.")
        
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        # self.options holds net_options and fit_options
        self.options = kwargs 

        self.options_valid = False

        self.check_options()

        # Initialize other attributes
        self._x_trn = None
        self._y_trn = None
        self._x_val = None
        self._y_val = None
        self.uqnet = None

        # Deferring construction of NN
        self.nnet = None
        

    def read_options_from_file(self, json_filename):
        """Read in options from given json file."""

        # Load JSON file 
        with open(json_filename, 'r') as json_file:
            self.options = json.load(json_file)
  
        # Convert 'wp_function_name' and 'wp_function_arg' into LayerFcn object
        if 'wp_function_name' in self.options:
            func_name = self.options['wp_function_name']
            # print('Instantiating weight parameterization function given')
            if (func_name == 'Poly' or func_name == 'NonPar') and not (self.options['wp_function_arg'] >= 0 and isinstance(self.options['wp_function_arg'], int)):
                raise ValueError(f"The LayerFcn function {self.options['wp_function_name']} requires a non-negative, integer argument 'wp_function_arg'.")
            if func_name == 'Poly':
                self.options['wp_function'] = Poly(self.options['wp_function_arg'])
                self.options.pop('wp_function_arg')
            if func_name == 'NonPar':
                self.options['wp_function'] = NonPar(self.options['wp_function_arg'])
                self.options.pop('wp_function_arg')
            
            if func_name == 'Cubic': self.options['wp_function'] = Cubic()
            if func_name == 'Quad': self.options['wp_function'] = Quad()
            if func_name == 'Lin': self.options['wp_function'] = Lin()
            if func_name == 'Const': self.options['wp_function'] = Const()

            self.options.pop('wp_function_name')

            if 'wp_function_arg' in self.options:
                raise ValueError(f"The LayerFcn functions Cubic, Quad, Lin, and Const do not take any arguments, but 'wp_function_arg' was provided.")

        self.check_options()


    def check_options(self):
        """Sanity check on current set of options, including instantiation of neural network, building,
            and evaluating. Throws error if invalid option.
        """
        self.options_valid = False

        expected_options = {
            # net_options for RNet:
            'wp_function': (LayerFcn, type(None)),  # Can be a LayerFcn object or None
            'indim': (int, type(None)),             # Can be an int or None
            'outdim': (int, type(None)),
            'layer_pre': bool,
            'layer_post': bool,
            'biasorno': bool,
            'nonlin': bool,
            'mlp': bool,
            'final_layer': (str, type(None)),       # Can be a string or None
            'device': (torch.device, str),          
            'init_factor': float,
            'sum_dim': int,
            # fit_options for nn_vi:
            'datanoise': float,
            'lrate': float,
            'batch_size': (int, type(None)),
            'nepochs': int,
            # build() options:
            'verbose': bool,
            # evaluate() options:
            'msc': (int, type(None)), 
            'nsam': (int, type(None)), 
        }   

        # Iterate through options in self.options
        for option, value in self.options.items():
            # (1) Check if the option is an invalid option name
            if option not in expected_options:
                raise ValueError(f"Invalid or currently unsupported option: {option}")
        
            # Get the possible valid types for the given option
            valid_types = expected_options[option]

            # (2) Validate via isinstance
            if not isinstance(value, valid_types):
                raise ValueError(f"Option '{option}' must be of type {valid_types}, received {type(value)}.")

            # (3) Special check for positive values
            if not isinstance(value, bool) and isinstance(value, (int, float)) and value <= 0:
                raise ValueError(f"Option '{option}' must be a non-negative value.")
            
            # TODO: For values that are allowed to be negative/benefit from a range, add a dictionary for valid ranges or multiple options.

            # (4) Special check for final_layer with specific string types
            if option == 'final_layer':
                if value not in ['sum', 'exp', None]:
                    raise ValueError(f"Option '{option}' can only be 'exp', 'sum', or None.")

        self.options_valid = True
    
    def get_options(self, json_print=None):
        """Return options in a dictionary and, if provided, write options to a json file."""
        # TODO: QUINN returns options defaulted to / used -> return library with defaults
        
        # Write read-in options to new JSON file
        if json_print is not None:
            with open(json_print, 'w') as file:
                printing_options = self.options.copy()
                if 'wp_function' in printing_options:
                    printing_options['wp_function_name'] = type(printing_options['wp_function']).__name__
                    if printing_options['wp_function_name'] == 'Poly':
                        printing_options['wp_function_arg'] = printing_options['wp_function'].npar - 1
                    else:
                        printing_options['wp_function_arg'] = printing_options['wp_function'].npar
                    printing_options.pop('wp_function')
                json.dump(printing_options, file, indent=4)

        return self.options

    def set_validation_data(self, x_val, y_val):
        r"""Sets the validation data with input validation (optional). If not implemented,
        the fitting method will default to using the training set for validation.

        Args:
            x_val (np.ndarray): Validation input array.
            y_val (np.ndarray): Validation output array.

        Raises:
            ValueError: If x_val or y_val do not meet the required dimensions.
        """
        if not (isinstance(x_val, np.ndarray) and x_val.ndim == 2):
            raise ValueError("x_val must be a 2D numpy array.")
        if not (isinstance(y_val, np.ndarray) and y_val.ndim == 2 and y_val.shape[1] == 1):
            raise ValueError("y_val must be a 2D numpy array with a single column for scalar-valued outputs.")
        if x_val.shape[0] != y_val.shape[0]:
            raise ValueError("The number of samples in x_val and y_val must be the same.")

        self._x_val = x_val
        self._y_val = y_val

    def set_training_data(self, x_trn, y_trn):
        r"""Sets the training data with input validation.

        Args:
            x_trn (np.ndarray): Training input array of size (N,d).
            y_trn (np.ndarray): Training output array of size (N,o).

        Raises:
            ValueError: If x_trn or y_trn do not meet the required dimensions.
        """
        if not (isinstance(x_trn, np.ndarray) and x_trn.ndim == 2):
            raise ValueError("x_trn must be a 2D numpy array.")
        if not (isinstance(y_trn, np.ndarray) and y_trn.ndim == 2 and y_trn.shape[1] == 1):
            raise ValueError("y_trn must be a 2D numpy array with a single column for scalar-valued outputs.")
        if x_trn.shape[0] != y_trn.shape[0]:
            raise ValueError("The number of samples in x_trn and y_trn must be the same.")

        self._x_trn = x_trn
        self._y_trn = y_trn

    def instantiate_network(self):
        """Instantiate neural network with validated options."""
        # TODO: Make issue to figure out how to expand to multiple different network types; including check_options input validation

        self.check_options()
        if self.options_valid:
            # List of all possible net options for RNet construction
            possible_net_options = [
                'wp_function', 'indim', 'outdim', 'layer_pre', 'layer_post', 'biasorno',  'nonlin', 'mlp', 
                'final_layer',  'init_factor', 'sum_dim'
                ]
            net_options = {}

            # Iterate through self.options, add values to net_options
            for option, value in self.options.items():
                if option in possible_net_options:
                    net_options[option] = value

            if self.type == 'RNet':
                self.nnet = RNet(self.n_nodes, self.n_layers, **net_options)


    def update_options(self, new_options):
        """Update only the options for building/fitting and evaluating of neural network."""
        possible_fit_options = ['datanoise', 'lrate', 'batch_size', 'nepochs', 'msc', 'nsam', 'verbose']

        # Check for invalid options
        for option in new_options:
            if option not in possible_fit_options:
                raise ValueError(f"Invalid fit or evaluate option: {option}. Please note: cannot update network options through build.")

        # Update self.options with the new fit_options from kwargs
        for option in possible_fit_options:
            if option in new_options:
                self.options[option] = new_options[option]

        # Validate the updated options
        self.check_options()


    def nn_vi(self):
        """Performs variational inference fitting and training of the neural network using the NN_VI wrapper class.

        Args:
            val (tuple): x,y tuple of validation points. Default uses the training set for validation.
        """
        # List of all possible fit options for variational inference 
        possible_fit_options = ['datanoise', 'lrate', 'batch_size', 'nsam', 'nepochs']

        fit_options = {}

        # Iterate through self.options, add values to fit_options
        for option, value in self.options.items():
            if option in possible_fit_options:
                fit_options[option] = value

        # Instantiate variational inference object 
        self.uqnet = NN_VI(self.nnet, verbose=self.options.get('verbose', True))

        # print('Fitting neural network with variational inference')
        # Call fit function, pass in fit_options for training of the neural network
        if self._x_val is None and self._y_val is None:
            self.uqnet.fit(self._x_trn, self._y_trn, **fit_options)
        else:
            self.uqnet.fit(self._x_trn, self._y_trn, val=[self._x_val,self._y_val], 
                        **fit_options)


    def build(self, **kwargs):
        """Builds the model with training data by calling the correct fitting method for model training.

        Args:
            fit_options (dict, optional): Options/training parameters for model fitting.
        """
        if self.nnet is None:
            self.instantiate_network()
        
        if kwargs is not None:
            self.update_options(kwargs)

        # Confirm that training data has been set
        if self._x_trn is None or self._y_trn is None:
            raise RuntimeError("Training data must be set using set_training_data() before calling build().")

        # Fit the neural network with VI
        self.nn_vi()


    def evaluate(self, x_eval, **kwargs):
        r"""Generates samples of predictions.

        Args:
            x_eval (np.ndarray): 2d array of size `(N,d)` as input data for evaluation. Can also be a single sample as input.
            nens (int, optional): Number of samples requested, M.
            msc (int, optional): Prediction mode: 0 (mean-only), 1 (mean and variance), or 2 (mean, variance and covariance). Defaults to 0.
            nens = nsam (int, optional): Number of samples requested, `M`.
        Returns:
            dictionary: Dictionary with samples of predictions.
            tuple(np.ndarray, np.ndarray, np.ndarray): triple of Mean (array of size `(N, o)`), 
            Variance (array of size `(N, o)` or None),
            Covariance (array of size `(N, N, o)` or None).

        """
        if self.nnet is None:
            raise Exception("The neural network must be instantiated with build() before calling evaluate().")
        
        if kwargs is not None:
            self.update_options(kwargs)

        # Call predict function for mean, variance, and covariance. If a certain value (ex] covariance) is not requested, function returns None.
        y_eval, y_var, y_cov = self.uqnet.predict_mom_sample(x_eval, msc = self.options.get('msc', 2), nsam = self.options.get('nsam', 1000))
        
        # Return dictionary with the samples of predictions and uncertainties
        return {
            'Y_eval': y_eval,
            'Y_eval_var': y_var,
            'Y_eval_cov': y_cov
        }
    
