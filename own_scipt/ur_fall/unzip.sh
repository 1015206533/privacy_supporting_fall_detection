#!/bin/bash

dirc='./'
if [ $# -gt 0 ]; then
	dirc=$1
	echo "file dir: ${dirc}"
fi

for file in `ls $dirc` 
do
	echo $file
	unzip $file
done
