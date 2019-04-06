#!/bin/bash

COUNTER=0
while true
do
    echo Iteration number $COUNTER
	python -u run_one.py || true
	let COUNTER=COUNTER+1
done
