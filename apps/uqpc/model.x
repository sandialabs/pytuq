#!/bin/bash

INPUT_FILE=$1 
OUTPUT_FILE=$2

# Run the black-box model,
# (a) Test function with 3 inputs and 5 outputs
awk '{print exp($1), $3*log($1**2)+$1+($2**2)*(1-exp(-$3**2)), $1+$3**2, $2+$3, $1*$3}' $1 > $2

# (b) Entry-wise exponential, same number of inputs and outputs
#awk '{for (i=1; i<=NF; i++) { printf("%lg ",exp($i)) }; printf("\n")}' $1 > $2


# # (c) A three parameter time-dependent model, i.e. as many outputs as there are time grid points
# # Warning: this will generate a lot of figures after the whole workflow is done.
# NGRID=50
# echo -n"" > $2
# for ((tgrid=1;tgrid<=$NGRID;tgrid++)); do
#     awk '{print $3*$1**2+$2*exp((tgr/ngr)*$3)}' tgr=$tgrid ngr=$NGRID $1 > tmp
#     paste $2 tmp > tmpp; mv tmpp $2; rm -f tmp
# done
# # Output names are time grid points
# rm -rf outnames.txt
# for ((i=1;i<=$NGRID;i++)); do
#     echo "$i" >> outnames.txt
# done