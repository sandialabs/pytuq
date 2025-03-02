#!/bin/bash -e

#SDIR=`dirname "$0"`


PCTYPE="HG"
ORDER=3

${KLPC}/apps/pc_prep.py mvn mm cc #mean.txt cov.txt
${KLPC}/apps/pc_sam.py pcf.txt $PCTYPE

awk '{print $1+$2, $1**2+$3, $3**3}' qsam.txt > ysam.txt

${KLPC}/apps/pc_fit.py $PCTYPE $ORDER