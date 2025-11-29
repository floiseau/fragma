# fragma

[![GitHub](https://img.shields.io/badge/GitHub-fragma-blue)](https://github.com/floiseau/fragma)
[![PyPI](https://img.shields.io/badge/PyPI-fragma-green)](https://pypi.org/project/fragma/)
[![Documentation](https://img.shields.io/badge/Docs-fragma-blueviolet)](https://floiseau.github.io/fragma)

A **finite element solver** for simulating crack propagation using the **phase-field approach to fracture**.
Built on top of [`fenicsx`](https://fenicsproject.org/), `fragma` supports advanced features like:
- Crack propagation in anisotropic media
- Indirect load control using path-following methods

The full documentation is available at: [https://floiseau.github.io/fragma](https://floiseau.github.io/fragma).

---
## 📦 Installation

To run [`fragma`](https://github.com/floiseau/fragma), install the required Python modules in a dedicated environment:

1. Create and activate a dedicated environment:
```bash
conda create -n fragma
conda activate fragma
```
2. Install FEniCSx (with GMSH):
```bash
conda install -c conda-forge fenics-dolfinx=0.10 pyvista mpich gmsh # Linux and macOS
conda install -c conda-forge fenics-dolfinx=0.10 pyvista pyamg gmsh # Windows
```
3. Install `fragma`:
```bash
pip install .       # If you cloned the repo
pip install fragma  # If you want to install from pypi
```

---
## Quick Start

Test `fragma` with ready-to-run examples:

```
cd examples/XX_example_name
./run.sh
```

Visualize results using [Paraview](https://www.paraview.org/) or [PyVista](https://docs.pyvista.org/).

---
## Usage

### Requirements
- A **GMSH mesh file** (`mesh.geo`)
- A **`parameters.toml`** configuration file (see [`examples`](https://github.com/floiseau/fragma/tree/main/examples))

### Running a Simulation
1. Navigate to your simulation directory.
2. Activate the environment:
    ```bash
    conda activate fragma
    ```
3. Run the solver:
    ```bash
    fragma
    ```

**Note:**
- Use `OMP_NUM_THREADS=1` on some Linux distributions to control thread usage:
    ```bash
    OMP_NUM_THREADS=1 python path/to/repo/fragma/main.py
    ```

### Outputs
- Results are saved in the `results` directory:
  - **VTK files** (e.g., `Displacement.pvd`) for field visualization
  - **CSV files** (e.g., `probes.csv`) for scalar outputs

