.. _quickstart:
 
==========
Quickstart
==========
 
This guide will help you get started with PyTUQ by walking through common use cases and workflows.
 
Basic Workflow
--------------
 
A typical uncertainty quantification workflow in PyTUQ consists of:
 
1. **Define the problem**: Set up input parameter distributions and ranges
2. **Generate samples**: Create samples from the input distributions
3. **Build a surrogate model**: Use polynomial chaos, Gaussian processes, or neural networks
4. **Perform analysis**: Compute sensitivities, propagate uncertainty, or calibrate parameters
 
Example 1: Polynomial Chaos Expansion
--------------------------------------
 
This example shows how to build a polynomial chaos expansion (PCE) surrogate model.
 
.. code-block:: python
 
    import numpy as np
    from pytuq.rv.pcrv import PCRV
    from pytuq.utils.mindex import get_mi
    from pytuq.utils.maps import scale01ToDom
 
    # Define problem parameters
    dim = 2          # Number of input dimensions
    order = 3        # Polynomial order
    N = 100          # Number of training samples
 
    # Define input domain
    domain = np.array([[-1., 1.],   # x1 range
                       [-1., 1.]])  # x2 range
 
    # Generate random samples in [0,1] and map to domain
    xi = np.random.rand(N, dim)
    x = scale01ToDom(xi, domain)
 
    # Define your computational model (example: simple function)
    def model(x):
        return np.sin(x[:, 0]) + 0.5 * x[:, 1]**2
 
    # Evaluate model at sample points
    y = model(x)
 
    # Build multiindex for polynomial basis
    mindex = get_mi(order, dim)
 
    # Create PC random variable with uniform distributions
    # 'LU' indicates Legendre polynomials for Uniform distributions
    pcrv = PCRV(1, dim, 'LU', mi=mindex)
 
    # Evaluate polynomial bases at sample points
    basis_matrix = pcrv.evalBases(x, 0)
 
    # Fit PC coefficients using least squares
    pc_coefficients = np.linalg.lstsq(basis_matrix, y, rcond=None)[0]
 
    # Store coefficients in PCRV object
    pcrv.setMiCfs([mindex], [pc_coefficients])
 
    # Make predictions at new points
    x_new = np.random.rand(10, dim)
    x_new = scale01ToDom(x_new, domain)
    basis_new = pcrv.evalBases(x_new, 0)
    y_pred = basis_new @ pc_coefficients
 
    print(f"Predictions at new points: {y_pred}")
 
Example 2: Global Sensitivity Analysis
---------------------------------------
 
Once you have a PCE surrogate, you can compute sensitivity indices:
 
.. code-block:: python
 
    from pytuq.rv.pcrv import PCRV
    from pytuq.utils.mindex import get_mi
    import numpy as np
 
    # Assuming you have built a PCE model as in Example 1
    dim = 2
    order = 3
    mindex = get_mi(order, dim)
    pcrv = PCRV(1, dim, 'LU', mi=mindex)
 
    # Set coefficients (from previous fitting)
    # pc_coefficients would come from your model fitting
    pc_coefficients = np.random.randn(mindex.shape[0])  # Example only
    pcrv.setMiCfs([mindex], [pc_coefficients])
 
    # Compute main and total sensitivity indices
    main_sens = pcrv.computeMainSens()
    total_sens = pcrv.computeTotSens()
 
    print("Main sensitivity indices:", main_sens[0])
    print("Total sensitivity indices:", total_sens[0])
 
    # Compute joint sensitivities
    joint_sens = pcrv.computeJointSens()
    print("Joint sensitivity indices:", joint_sens[0])
 
Example 3: Bayesian Compressive Sensing
----------------------------------------
 
Use Bayesian Compressive Sensing (BCS) to automatically select relevant polynomial terms:
 
.. code-block:: python
 
    import numpy as np
    from pytuq.rv.pcrv import PCRV
    from pytuq.utils.mindex import get_mi
    from pytuq.lreg.bcs import bcs
    from pytuq.utils.maps import scale01ToDom
 
    # Problem setup
    dim = 2
    order = 5
    N = 50
 
    # Generate data
    domain = np.array([[-1., 1.], [-1., 1.]])
    x = scale01ToDom(np.random.rand(N, dim), domain)
 
    # Simple quadratic model
    y = x[:, 0]**2 + 2*x[:, 1]
    y += 0.01 * np.random.randn(N)  # Add noise
 
    # Build full polynomial basis
    mindex = get_mi(order, dim)
    pcrv = PCRV(1, dim, 'LU', mi=mindex)
    basis_matrix = pcrv.evalBases(x, 0)
 
    # Perform BCS fit
    lreg = bcs(eta=1.e-10)  # eta controls sparsity
    lreg.fita(basis_matrix, y)
 
    # Show which terms were selected
    print(f"Number of terms selected: {len(lreg.used)}")
    print(f"Selected indices: {lreg.used}")
    print(f"Coefficients: {lreg.cf}")
 
    # Make predictions with selected terms
    x_test = scale01ToDom(np.random.rand(5, dim), domain)
    basis_test = pcrv.evalBases(x_test, 0)
    y_pred, y_var, _ = lreg.predicta(basis_test[:, lreg.used], msc=1)
 
    print(f"Predictions: {y_pred}")
    print(f"Prediction variance: {y_var}")
 
Example 4: Gaussian Process Regression
---------------------------------------
 
Build a Gaussian process surrogate model:
 
.. code-block:: python
 
    import numpy as np
    from pytuq.gp.gp import gp
 
    # Generate training data
    dim = 1
    N = 20
    x_train = np.random.rand(N, dim) * 2 * np.pi
    y_train = np.sin(x_train).flatten() + 0.1 * np.random.randn(N)
 
    # Create GP object
    gp_model = gp(x_train, y_train)
 
    # Fit hyperparameters
    gp_model.fit()
 
    # Make predictions
    x_test = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
    y_pred, y_var = gp_model.predict(x_test)
 
    print(f"Mean prediction shape: {y_pred.shape}")
    print(f"Variance shape: {y_var.shape}")
 
Common Use Cases
----------------
 
**Uncertainty Propagation**
 
Use PCE or GP surrogates to propagate input uncertainties through your model:
 
1. Build a surrogate model from training data
2. Sample from input distributions
3. Evaluate the surrogate to get output distribution
 
**Parameter Calibration**
 
Use MCMC methods to calibrate model parameters:
 
.. code-block:: python
 
    from pytuq.mcmc.mcmc import MCMC
 
    # Define log-likelihood and log-prior functions
    def log_likelihood(params):
        # Compare model output to data
        residual = model(params) - data
        return -0.5 * np.sum(residual**2 / sigma**2)
 
    def log_prior(params):
        # Define prior distributions
        if np.all((params > lower_bounds) & (params < upper_bounds)):
            return 0.0
        return -np.inf
 
    # Create MCMC object and run
    mcmc = MCMC(log_likelihood, log_prior, initial_params)
    chain = mcmc.run(nsteps=10000)
 
**Sensitivity Analysis**
 
Identify which input parameters most affect your output:
 
1. Build a PCE surrogate
2. Compute Sobol indices using ``computeMainSens()`` and ``computeTotSens()``
3. Visualize results to identify important parameters
 
Next Steps
----------
 
- Explore the :doc:`auto_examples/index` for detailed examples
- Check the :doc:`API Reference <../autoapi/index>` for complete documentation
- See the :doc:`about` page for an overview of all capabilities
 
Tips and Best Practices
------------------------
 
1. **Start simple**: Begin with low-dimensional problems and low polynomial orders
2. **Use BCS for high-order PCE**: Bayesian compressive sensing automatically selects relevant terms
3. **Check convergence**: For MCMC, always check chain convergence and mixing
4. **Validate surrogates**: Compare surrogate predictions against true model evaluations
5. **Scale inputs**: Normalize inputs to [-1, 1] or [0, 1] for better numerical stability
