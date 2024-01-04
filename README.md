# fragma

`fragma` is a finite element solver to simulate crack propagation in anisotropic media using the phase-field approach to fracture.
This solver uses `fenicsx`.

## Installation
To run `fragma`, different python modules must be installed first.
One can apply the following commands to install those modules inside an environment.
```shell
$ conda create -n fenicsx-env
$ conda activate fenicsx-env
$ conda install -c conda-forge fenics-dolfinx mpich pyvista python-gmsh mpi4py petsc4py
```
In practice, I used `micromamba` instead of `conda` to manage the environment.


## Usage
In order to run, `fragma` needs a GMSH mesh file and the parameter file (`parameters.toml`).
The content of the parameter file is described below, and examples are given in the `examples` directory.

Once both files are provided, different steps are necessary to run it.
1. Go to the directory containing the parameter file `parameters.toml`.
2. Activate the `fenicsx-env` environment (`conda activate fenicsx-env` or `micromamba activate fenicsx-env`).
3. Run `python path/to/repo/fragma/main.py`

*It is also possible to run* `fragma` *directly in the environment using* `conda run -n fenicsx-env python path/to/repo/fragma/main.py` *(or the same comment with*`micromamba`*). Also, note that the examples contain a* `run.sh` *file, which generates the mesh and runs* `fragma`*.*

`fragma` will then generate a results directory containing the `results.xdmf` file (and the `results.h5` containing the results).
The `results.xdmf` file can opened with Paraview to visualize the simulation results.

It is also possible to use the `run.sh` files that are provided in the example to run `fragma`.

*Remark: When using `conda`, one needs to update the `run.sh` files in the example by replacing `micromamba` with `conda`.*

## Content of input files
TODO

## Content of output files
TODO

## Ressources used to make the code

- [FEniCSx tutorial](https://jsdokken.com/dolfinx-tutorial/index.html)
- [DOLFINx documentation](https://docs.fenicsproject.org/dolfinx/main/python/index.html)
- [NewFrac FEniCSx Training](https://newfrac.gitlab.io/newfrac-fenicsx-training/04-phase-field/phase-field.html#time-stepping-solving-a-quasi-static-problem)
- [https://bitbucket.org/bin-mech/anisotropic-gradient-damage](https://bitbucket.org/bin-mech/anisotropic-gradient-damage)
- [https://github.com/jaedong2019/Monotonic_loading](https://github.com/jaedong2019/Monotonic_loading)
