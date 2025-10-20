#!/bin/bash
#=====================================================================================

# Given a Nxd matrix of samples, compute dx2 ranges
# Example: getrange.x samples.dat > ranges.dat

shopt -s expand_aliases
alias awk="awk -v OFMT='%.15e'"

if [ $# -lt 1 ]; then
    echo "Number of arguments can not be less than 1"
    echo "Syntax: $0 <filename> [<cushion_fraction>]"
    exit
elif [ $# -gt 2 ]; then
    echo "Number of arguments can not be greater than 2"
    echo "Syntax: $0 <filename> [<cushion_fraction>]"
    exit
elif [ $# -eq 1 ]; then
    fr=0.0
elif [ $# -eq 2 ]; then
    fr=$2
fi

filename=$1

DIM=`awk 'END{print NF}' $filename`

for (( COL=1; COL<=$DIM ; COL++ )); do

awk 'BEGIN {
valmin=1e+100;
valmax=-1e+100;
line=1;
}
{
if( $(col) < valmin )
{
  valmin=$(col);
}
if( $(col) > valmax )
{
  valmax=$(col);
}
}
END{
print valmin-fr*(valmax-valmin), valmax+fr*(valmax-valmin)
}' col=$COL $filename
done
