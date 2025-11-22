A collection of scripts that propagate input parameter uncertainties to output
via PC expansions.

As a special, and most commonly used, case the scripts can construct a PC
surrogate for a multi-output computational model. The latter is as a black box
simulation code. The workflow also provides tools for global sensitivity
analysis of the outputs of this black box model with respect to its input
parameters or input PC germs.


================================================================================
Files

- workflow_uqpc.x: an example workflow script showing how to use various
  components to build a PC surrogate for a black-box model.

- uq_pc.py: the main script, see ./up_pc.py -h for options

- model.x   :  An example black-box model, mapping inputs to outputs.

- plot.py   :  Plotting after surrogate construction, reading the pickle file
  results.pk produced by uq_pc.py.
