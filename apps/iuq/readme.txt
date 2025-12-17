A collection of scripts that perform parameter inference using Markov chain Monte-Carlo sampling 
and employing polynomial chaos (PC) surrogate model constructed in the sibling workflow uqpc/.

Besides conventional inference, embedded model error inference is implemented as well.
The workflow, including postprocessing (plotting) scripts can run out-of-the-box, but
the intention is for this to be a demo: the user should take it locally and adjust to 
specifics of their problem as there are too many knobs to tune typically.

================================================================================
Files

- workflow_iuq.x        : an example workflow script going through the necessary python scripts.

- run_infer.py          : the main script, performing PC-surrogate based Bayesian inference.

- create_data_truepc.py : example of a synthetic data generation.
- prep_data.x           : another example of a synthetic data generation.

- postp_infer.py        : parsing the results and plotting many png files.