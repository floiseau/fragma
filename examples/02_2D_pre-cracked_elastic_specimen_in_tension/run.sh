#!/usr/bin/env bash

# Generate the mesh
gmsh -2 mesh.geo

# Run the simulation
micromamba run -n fenicsx-env python ../fragma/fragma/main.py

