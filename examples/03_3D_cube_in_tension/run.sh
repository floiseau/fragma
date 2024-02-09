#!/usr/bin/env bash

# Generate the mesh
gmsh -3 mesh.geo

# Run the simulation
OMP_NUM_THREADS=1 python ../../fragma/main.py

