#!/bin/bash -e




PUQAPPS=`dirname "$0"`


PCTYPE="HG"
ORDER=3
NSAM=111

echo "3.0" > mean.txt
echo "-2.1" >> mean.txt
echo "1.5 0.4" > cov.txt
echo "0.4 2.0" >> cov.txt



${PUQAPPS}/pc_prep.py -f mvn -i mean.txt -c cov.txt
${PUQAPPS}/plot_cov.py -m mean.txt -c cov.txt
${PUQAPPS}/pc_sam.py -f pcf.txt -t $PCTYPE -n $NSAM

awk '{print $1+$2, -$1**2+$2, $2**3}' qsam.txt > ysam.txt

${PUQAPPS}/pc_fit.py -x qsam.txt -y ysam.txt -c $PCTYPE -o $ORDER