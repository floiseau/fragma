#!/usr/bin/env bash

# Generate the mesh
gmsh -2 mesh.geo

# Run the simulation
OMP_NUM_THREADS=1 python ../../fragma/main.py

