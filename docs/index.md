# Overview

## Description

[`fragma`](https://github.com/floiseau/fragma) is a 2D **finite element solver** for simulating crack propagation using the **phase-field approach to fracture**.
It is built on top of [`fenicsx`](https://fenicsproject.org/).

It has several non-classic features :

- crack propagation in anistropic media,
- indirect load control using path-following methods.

---

## Installation

To run [`fragma`](https://github.com/floiseau/fragma), install the required Python modules in a dedicated environment:

1. Create and activate a dedicated environment:

        conda create -n fragma
        conda activate fragma

2. Install FEniCSx (with GMSH):

        conda install -c conda-forge fenics-dolfinx=0.10 pyvista mpich gmsh # Linux and macOS
        conda install -c conda-forge fenics-dolfinx=0.10 pyvista pyamg gmsh # Windows

3. Install `fragma`:


        pip install .       # If you cloned the repo
        pip install fragma  # If you want to install from pypi



---

## Hands-On: Test with Examples

The [`examples`](https://github.com/floiseau/fragma/tree/main/examples) directory contains ready-to-run scripts and parameter files.
Just follow these steps:

1. Navigate to the [`examples`](https://github.com/floiseau/fragma/tree/main/examples) directory.
2. Run the provided `run.sh` script:

        cd examples
        ./run.sh

3. Visualize the results in Paraview.

---

## Usage

### Requirements
To run a simulation, the following files are required in the simulation directory:

- A **GMSH mesh file** : `mesh.geo`
- A **`parameters.toml`** configuration file (see the [`examples`](https://github.com/floiseau/fragma/tree/main/examples) directory for reference)

### Running the Solver
1. Navigate to the simulation directory containing (the mesh and the `parameters.toml` file).
2. Activate the `fragma` environment:

        conda activate fragma

3. Run the solver:

        python path/to/repo/fragma/main.py

!!! Notes
    - The [`examples`](https://github.com/floiseau/fragma/tree/main/examples) include a `run.sh` script to automate mesh generation and execution.
    - On some Linux distributions, set `OMP_NUM_THREADS=1` to prevent FEniCSx from using all available threads:

            OMP_NUM_THREADS=1 python path/to/repo/fragma/main.py

### Outputs

Results are saved in the `results` subdirectory.
Two types of file are exported:

- The solution fields (displacement, phase-field, etc.) are saved in `VTK` files (*e.g.*, `Displacement.pvd`).
- The scalar outputs (energies, displacement/force measures, etc.) are saved in the `probes.csv` CSV file.

!!! Vizualization tools
    To open the `VTK` files, vizualization tools such as [Paraview](https://www.paraview.org/) or [pyvsita](https://docs.pyvista.org/index.html) can be employed.
