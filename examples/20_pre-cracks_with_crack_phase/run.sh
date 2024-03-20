#!/usr/bin/env bash

# Generate the mesh
gmsh -2 mesh.geo

# Remove previous results
rm -r results

# Run the simulation
OMP_NUM_THREADS=1 python ~/ownCloud/codes/fragma/fragma/main.py

