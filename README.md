# fragma

`fragma` is a finite element solver to simulate crack propagation in anisotropic media using the phase-field approach to fracture.
This solver uses `fenicsx`.

## Installation
To run `fragma`, different python modules must be installed first.
One can apply the following commands to install those modules inside an environment.
```shell
$ conda create -n fragma
$ conda activate fragma
$ conda install -c conda-forge fenics-dolfinx mpich pyvista python-gmsh mpi4py petsc4py scipy sympy
```
Note that it is recommanded to use [libmamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) for a faster environment creation.

## Usage
In order to run, `fragma` needs a GMSH mesh file and the parameter file (`parameters.toml`).
The content of the parameter file is described below, and examples are given in the `examples` directory.

Once both files are provided, different steps are necessary to run it.
1. Go to the directory containing the parameter file `parameters.toml`.
2. Activate the `fragma` environment using the command: `conda activate fragma`.
3. Run `python path/to/repo/fragma/main.py`

*It is also possible to run* `fragma` *directly in the environment using* `conda run -n fragma python path/to/repo/fragma/main.py` . Also, note that the examples contain a* `run.sh` *file, which generates the mesh and runs* `fragma`*.*

`fragma` will then generate a results directory containing the VTK files.
The files names `quantity.pvd` (*e.g.*, `Displacement.pvd`) can be opened with Paraview to visualize the simulation results.

*Remark: In some Linux distribution, the environment variable `OMP_NUM_THREADS` is not set, leading FEniCSx solve the same problem on all the available cores. To prevent that, `OMP_NUM_THREADS=1` must be prepended to the command to run `fragma`.*

## Dashboard
To see intermediate results and track the progress of a simulation, a dashboard in included with `fragma`.
To use this dashboard, additional dependencies (`plotly`) are required.
To install them, run the following command.
```shell
$ conda activate fragma
$ conda install plotly
```
The files and instructions required to use the dashboard are available in the [tools/dashboard](tools/dashboard/) directory.

## Content of input files
TODO

## Content of output files
TODO

## Ressources used to make the code

- Generic 
    - [FEniCSx tutorial](https://jsdokken.com/dolfinx-tutorial/index.html)
    - [DOLFINx documentation](https://docs.fenicsproject.org/dolfinx/main/python/index.html)
    - [NewFrac FEniCSx Training](https://newfrac.gitlab.io/newfrac-fenicsx-training/04-phase-field/phase-field.html#time-stepping-solving-a-quasi-static-problem) (outdated)
- Implementations of phase-field fracture
    - [https://bitbucket.org/bin-mech/anisotropic-gradient-damage](https://bitbucket.org/bin-mech/anisotropic-gradient-damage) (legacy fenics)
    - [https://github.com/jaedong2019/Monotonic_loading](https://github.com/jaedong2019/Monotonic_loading) (legacy fenics)
- Numerical optimizations
    - Elasticity solver
        - [Fenicsx demo: Elasticity using algebraic multigrid](https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_elasticity.html)
    - Over-relaxation
        - Farrell, P., & Maurini, C. (2017). Linear and nonlinear solvers for variational phase-field models of brittle fracture. International Journal for Numerical Methods in Engineering, 109(5), 648–667. [https://doi.org/10.1002/nme.5300](https://doi.org/10.1002/nme.5300)


