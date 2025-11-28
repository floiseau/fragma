# Parameters file

The `parameters.toml` file configures simulations in `fragma`.
This section gives a broad (and non-exhaustive) overview of its organization.
It is organized into sections, each defining a specific aspect of the simulation.
For practical examples, refer to [`examples`](https://github.com/floiseau/fragma/tree/main/examples).

---

### `[model]`
- `name`: Type of simulation (`"elasticity"`, `"fracture"`).
- `dim`: Dimensionality of the problem (`2`) , `3` works only with elasticity.
- `2D_assumption` (if 2D): Assumption for 2D problems (`"plane_strain"`, `"plane_stress"`).
- `model` (if fracture): Phase-field model type (e.g., `"AT1"`, `"AT2"`).

---

### `[mesh]`
- `msh_file`: Name of the mesh file (e.g., `"mesh.msh"`).
- `[mesh.physical_groups]`: Physical groups defined in the mesh, used for boundary conditions or material properties.

        [mesh.physical_groups]
        left  = 5
        right = 6

    Physical groups can also be set as crack (α=1) or non-crackable (α=0).

        [mesh.physical_groups]
        non-crackable_1 = 9
        non-crackable_2 = 10
        crack = 4

---

### `[mechanical]`
- Material properties:
  - `E`: Young’s modulus (in Pa).
  - `nu`: Poisson’s ratio.
  - `Gc` (if fracture): Critical energy release rate (in J/m²).
  - `ell` (if fracture): Regularization length (in m).

---

### `[loading]`
- Boundary conditions:
    - `[loading.u_imp_max]`: Imposed displacements.

            [loading.u_imp_max]
            left  = [0, 0]

    - `[loading.f_imp_max]`: Imposed forces.

            [loading.f_imp_max]
            right = [0.1, nan]

- Path-following:

        [loading]
        constraint = "max_inc_undamaged_elastic_energy"
        dtau = 2e5

    The constraint corresponds to the name of the constraint equation and `dtau` to the step size.
    The available constraint are : `"max_strain_inc_outside_crack"` (recommended), `"load_factor_inc"` (no path-following), `"max_strain_inc"`, `"nodal_disp_inc"`, `"local_arc_length"`, `"crack_driving_variable"`, and `"maxt_crack_driving_variable"`.

!!! Note
    `nan` in the imposed displacement vector means that no displacement is imposed.

---

### `[numerical]`
- Solver parameters:
  - `utol`: Displacement tolerance.
  - `atol` (if fracture): Phase-field tolerance.
  - `max_iter` (if fracture): Maximum number of iterations in alternate minimization.
  - `alpha_res` (if fracture): Residual tolerance for non-linear solution of the phase-field.
  - `omega` (optional): Over-relaxation parameter.

---

### `[end]`
- Termination criterion:
  - `criterion`: `"t"` (time-based), `"elastic_energy_drop"`.
  - `t_max`: Maximum simulation step.
  - `drop`: Drop in elastic energy (E(t) / max_{t’≤t} E(t’) < drop).

---

### `[postprocess]` (optional)
- Post-processing options:
  - `[postprocess.probes]`: Coordinates for displacement probes.

    [postprocess.probes]
    displacement = [
        [0, 0.5, 0],
        [10, 0.5, 0],
    ]

---

### Example File

    [model]
    name = "fracture"
    dim = 2
    2D_assumption = "plane_stress"
    model = "AT1"

    [mesh]
    msh_file = "mesh.msh"

    [mesh.physical_groups]
    bot = 37
    top = 38
    crack = 39

    [mechanical]
    E = 3e9
    nu = 0.4
    Gc = 380
    ell = 0.1

    [numerical]
    utol = 1e-6
    atol = 1e-5
    max_iter = 1000
    alpha_res = 1e-12

    [loading]
    [loading.u_imp_max]
    bot = [0, -5e-4]
    top = [0,  5e-4]

    [end]
    criterion = "t"
    t_max = 50

---

### Key Notes
- **Physical Groups**: Must match those defined in the mesh file.
- **Optional Sections**: Some sections (e.g., `[numerical]`, `[postprocess]`) are optional.
