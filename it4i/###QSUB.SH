#!/bin/bash

rm -f *.o
rm -f CAE.*
echo -ne "qdel " > "#QDEL.SH"
ident=`qsub ./submit.sh $@`
echo $ident >> "#QDEL.SH"
chmod +x "#QDEL.SH"
echo -ne "qstat -a " > "##QSTAT.SH"
ident="${ident/'[]'/[1-6]}"
echo $ident >> "##QSTAT.SH"
chmod +x "##QSTAT.SH"
