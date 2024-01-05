#!/usr/bin/env bash

# Iterate through each directory
for dir in ./*/; do
    cd $dir
    ./run.sh
    cd ..
done
