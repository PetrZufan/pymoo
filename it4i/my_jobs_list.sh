#!/bin/bash

LOGIN=zufan

for ident in `qstat -u ${LOGIN} | grep ${LOGIN} | awk -F ' ' '{ print $1 }'`
do
	ident="${ident/'[]'/[1-6]}"
	qstat -a $ident | grep ${LOGIN}
done
