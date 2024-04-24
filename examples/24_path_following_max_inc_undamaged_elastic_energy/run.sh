#!/usr/bin/env bash

# Remove previous results
rm -r results

# Run the simulation
OMP_NUM_THREADS=1 python ../../fragma/main.py

