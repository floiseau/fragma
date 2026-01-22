#!/usr/bin/env bash

# Iterate through each directory
for dir in ./*/; do
    # Remove *.msh and *.db files
    rm $dir/*.msh
    rm $dir/*.db
    # Remove the "results" subdirectory
    rm -rf "$dir/results"
done
