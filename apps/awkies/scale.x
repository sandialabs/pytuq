#!/bin/bash -e
# Scales matrix data to or from a given parameter domain to [-1,1]^d
# scale.x <input> <to or from> <domain> <output>

shopt -s expand_aliases
alias awk="awk -v OFMT='%.15e'"

IN_FILE=$1
TO_FROM=$2
DOM_FILE=$3
OUT_FILE=$4

N=`awk 'END{print NR}' $IN_FILE`
D=`awk 'END{print NR}' $DOM_FILE`

DD=`awk 'END{print NF}' $IN_FILE`

## check that DD=D

echo "" > $OUT_FILE
for (( i=1; i<=$D ; i++ )); do
A=`awk 'NR==i{print $1}' i=$i $DOM_FILE`
B=`awk 'NR==i{print $2}' i=$i $DOM_FILE`

if [ "$TO_FROM" = "from" ]; then
    awk '{print -1.+2.*($i-a)/(b-a)}' i=$i a=$A b=$B $IN_FILE > tmp
elif [ "$TO_FROM" = "to" ]; then
    awk '{print (a+b)/2.+$i*(b-a)/2.}' i=$i a=$A b=$B $IN_FILE > tmp
else
    echo "Second argument has to be 'to' or 'from'"
    exit
fi
paste $OUT_FILE tmp > tmpp; mv tmpp $OUT_FILE

done
