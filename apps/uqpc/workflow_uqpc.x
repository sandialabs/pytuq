#!/bin/bash -e
#=====================================================================================

# Script location
export UQPC=`dirname "$0"`
export PUQAPPS=$UQPC/..


###############################
##    Setup the problem      ##
###############################


## Four simple options for uncertaininput parameter setup. 
## Uncomment one of them. 

## (a) Given mean and standard deviation of each normal random parameter
echo "1 0.3 " > param_margpc.txt
echo "3 0.1" >> param_margpc.txt
echo "1 0.5" >> param_margpc.txt
PC_TYPE=HG # Hermite-Gaussian PC
INPC_ORDER=1
# Creates input PC coefficient file pcf.txt (will have lots of zeros since we assume independent inputs)
${PUQAPPS}/pc_prep.py -f marg -i param_margpc.txt -p ${INPC_ORDER}

# ## (b) Given mean and half-width of each uniform random parameter
# echo "1 0.3 " > param_margpc.txt
# echo "3 0.1" >> param_margpc.txt
# echo "1 0.5" >> param_margpc.txt
# PC_TYPE=LU # Legendre-Uniform PC
# INPC_ORDER=1
# # Creates input PC coefficient file pcf.txt (will have lots of zeros since we assume independent inputs)
# ${PUQAPPS}/pc_prep.py -f marg -i param_margpc.txt -p ${INPC_ORDER}

# ## (c) Given mean and covariance of multivariate normal random parameter
# echo "3.0" > mean.txt
# echo "-2.1" >> mean.txt
# echo "1.5 0.4" > cov.txt
# echo "0.4 2.0" >> cov.txt
# PC_TYPE=HG # Hermite-Gaussian PC
# INPC_ORDER=1
# # Creates input PC coefficient file pcf.txt 
# ${PUQAPPS}/pc_prep.py -f mvn -i mean.txt -c cov.txt
# # Visualize covariance
# ${PUQAPPS}/plot_cov.py -m mean.txt -c cov.txt

# ## (d) Given samples of inputs in psam.txt (e.g. from a prior calibration study)
# ${PUQAPPS}/pc_prep.py -f sam -i psam.txt -p ${INPC_ORDER}
# PC_TYPE=HG # Hermite-Gaussian PC
# INPC_ORDER=3

# Number of samples requested
NTRN=111 # Training
NTST=33  # Testing	

# Extract dimensionality d (i.e. number of input parameters)
DIM=`awk 'NR==1{print NF}' pcf.txt`

# Output PC order
OUTPC_ORDER=3 

####################################
##  1-2. Online UQ with model.x   ##
####################################
## Run UQ with model.x in an online fashion (i.e. this is equivalent to the steps 1-2 below)
# Can uncomment this and comment out steps 1-2 below.
# ${UQPC}/uq_pc.py -r online_bb -c pcf.txt -x ${PC_TYPE} -d $DIM -o ${INPC_ORDER} -m anl -s rand -n $NTRN -v $NTST -t ${OUTPC_ORDER}


####################################
##  1-2. Pre-run input/output     ##
####################################

# # Quick and dirty equivalent if user already has the input/output pairs
# # Can uncomment this and comment out steps 1-2 below, IF input.txt and output.txt are available
# ln -sf input.txt pall.txt
# ln -sf output.txt yall.txt
# # Get ranges of inputs, with 10% 'cushion' from the dimension-wise extreme samples
# ${UQPC}/../awkies/getrange.x pall.txt 0.1 > param_range.txt
# # Scale the inputs
# ${UQPC}/../awkies/scale.x pall.txt from param_range.txt qall.txt

# # This is not ideal if input/output have strictly fewer or more rows than NTRN+NTST
# head -n$NTRN pall.txt > ptrain.txt
# head -n$NTRN qall.txt > qtrain.txt
# head -n$NTRN yall.txt > ytrain.txt

# tail -n$NTST pall.txt > ptest.txt
# tail -n$NTST qall.txt > qtest.txt
# tail -n$NTST yall.txt > ytest.txt

###################################
##  1-3. Online UQ with model()  ##
###################################
## Run UQ with model() function defined in uq_pc.py (i.e. this is equivalent to the steps 1-3 below)
# Can uncomment this and comment out steps 1-3 below.
# ${UQPC}/uq_pc.py -r online_example -c pcf.txt -x ${PC_TYPE} -d $DIM -o ${INPC_ORDER} -m anl -s rand -n $NTRN -v $NTST -t ${OUTPC_ORDER}



###############################
##  1. Prepare the inputs    ##
###############################

# Prepare inputs for the black-box model (use input PC to generate input samples for the model)
${PUQAPPS}/pc_sam.py -f pcf.txt -t ${PC_TYPE} -n $NTRN
mv psam.txt ptrain.txt; mv qsam.txt qtrain.txt
${PUQAPPS}/pc_sam.py -f pcf.txt -t ${PC_TYPE} -n $NTST
mv psam.txt ptest.txt; mv qsam.txt qtest.txt


# This creates files ptrain.txt, ptest.txt (train/test parameter inputs), qtrain.txt, qtest.txt (corresponding train/test stochastic PC inputs)


# Optionally can provide pnames.txt and outnames.txt with input parameter names and output QoI names
# Or delete them to use generic names
rm -f pnames.txt outnames.txt

################################
## 2. Run the black-box model ##
################################

# Run the black-box model, can be any model from R^d to R^o)
# ptrain.txt is N x d input matrix, each row is a parameter vector of size d
# ytrain.txt is N x o output matrix, each row is a output vector of size o 
${UQPC}/model.x ptrain.txt ytrain.txt
# Similar for testing
${UQPC}/model.x ptest.txt ytest.txt

# This creates files ytrain.txt, ytest.txt (train/test model outputs)

##############################
#  3. Build PC surrogate    ##
##############################

# Build surrogate for each output (in other words, build output PC)
${UQPC}/uq_pc.py -r offline -c pcf.txt -x ${PC_TYPE} -d $DIM -o ${INPC_ORDER} -m anl -s rand -n $NTRN -v $NTST -t ${OUTPC_ORDER}

# This creates files results.pk (Python pickle file encapsulating the results)


# Quick equivalent but here results are saved differently
# ${PUQAPPS}/pc_fit.py -x qtrain.txt -y ytrain.txt -c ${PC_TYPE} -o ${OUTPC_ORDER}
# This creates files pcrv.pk and lregs.pk (Python pickle file encapsulating the results)


###################################
##  4. Visualize the i/o data    ##
###################################

awk '{print "Training"}' ytrain.txt > labels.txt
awk '{print "Testing"}' ytest.txt >> labels.txt
cat ytrain.txt ytest.txt > yall.txt
cat ptrain.txt ptest.txt > pall.txt

${PUQAPPS}/plot_xx.py -x pall.txt -l labels.txt
${PUQAPPS}/plot_pcoord.py -x pall.txt -y yall.txt  -l labels.txt

${PUQAPPS}/plot_yx.py -x ptrain.txt -y ytrain.txt -c 3 -r 1
${PUQAPPS}/plot_yxx.py -x ptrain.txt -y ytrain.txt
${PUQAPPS}/plot_pdfs.py -p ptrain.txt; cp pdf_tri.png pdf_tri_inputs.png
${PUQAPPS}/plot_pdfs.py -p ytrain.txt; cp pdf_tri.png pdf_tri_outputs.png
${PUQAPPS}/plot_ens.py -y ytrain.txt

# A lot of .png files are created for visualizing the input/output data

################################
## 5. Postprocess the results ##
################################


# Plot model vs PC for each output
${UQPC}/plot.py dm training testing
# Plot model vs PC for each sample
${UQPC}/plot.py fit training testing

# Plot output pdfs
${UQPC}/plot.py pdf
# Plot output pdfs in joyplot format
${UQPC}/plot.py joy

# Plot main Sobol sensitivities (barplot)
${UQPC}/plot.py sens main
# Plot joint Sobol sensitivities (circular plot)
${UQPC}/plot.py jsens
# Plot matrix of Sobol sensitivities (rectangular plot)
${UQPC}/plot.py sensmat main

# Plot 1d slices of the PC surrogate
${UQPC}/plot.py 1d training testing
# Plot 2d slices of the PC surrogate
${UQPC}/plot.py 2d 

# This creates .png files for visualizing PC surrogate results and sensitivity analysis

