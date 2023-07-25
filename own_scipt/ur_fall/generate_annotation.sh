#!/bin/bash

dirc='./'
label=0
if [ $# -gt 1 ]; then
	dirc=$1
	label=$2
	echo "file dir: ${dirc}"
	echo "label: ${label}"
else
	echo "param error, need 2 param"
	exit 0
fi

for file in `ls $dirc` 
do
	file_name="${dirc}/${file}"
	num=`ls -l $file_name|wc -l`
	num=$[$num-1]
	echo $file_name $num $label
done
