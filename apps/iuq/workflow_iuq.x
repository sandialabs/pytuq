#!/bin/bash -e
#=====================================================================================

# Script location
export IUQ=`dirname "$0"`
export PUQAPPS=$UQPC/..



##############################
##    0. Prepare data       ##
##############################

# Remove, if any, true known parameter values
rm -f p_true.txt

### Prepare files ydata.txt and ydatavar.txt

## (a) From the actual PC model with some fixed coefficients, 
## so we know the true values of parameters to be inferred.
$IUQ/create_data_truepc.py -s 0.5 -e 7
## (b) External script to prepare data
# $IUQ/prep_data.x 

##############################
##    1. Run inference      ##
##############################

## (a) Classical
#$IUQ/run_infer.py
## (b) With embedded model error
$IUQ/run_infer.py -m


##############################
##    2. Postprocess        ##
##############################
# Optionally provide parameter names and output names in 
# pnames.txt and outnames.txt files
# otherwise the plotting will use generic names Par_ and Out_

# Generate inference results' figures
if [[ -f p_true.txt ]]; then
	$IUQ/postp_infer.py -c p_true.txt
else
	$IUQ/postp_infer.py 
fi