# Calculation of the SIF using domain integral

## Description
This example is dedicated to the post-process of the SIF ($K_{\mathrm{I}}$, $K_{\mathrm{II}}$ ,and the $T$-stress).
The calculation are based on the $I$-integral using a domain integral (using $G(\theta)$ method).

Here a Center-Crack Tensile (CCT) specimen is considered.
To obtain the SIFs under a loading, the `postprocess` section of the paramater file must be filled as shown below
```toml
crack_tip = [0.05, 0]
R_int = ...
R_ext = ...
crack_growth_angle = 0
```
where:
- `crack_tip` is the position of the crack tip,
- `R_int` is the interior of the circle (centered around the crack tip) in which the norm of the $\| \theta \|$ field is equal to 1, 
- `R_ext` is the exterior radius,
- `crack_growth_angle` is the angle $\phi$ of the virtual crack growth ($\theta = (\cos(\phi), \sin(\phi))$). The auxiliary fields are also rotated according the crack growth angle.
The transition where $\|\theta\|$ goes from 1 to 0 is between `R_int` and `R_ext`.

## Analytical solution
The obtained results are compared to the results from  Smith *et al.* (2001).

In our case, we have
```math
\sigma = 1
, \quad
\alpha = 0
```
where all quantities are expressed in SI unit.

## Running the example

In order to run this exemple, the following commands should be executed.
```shell
conda activate fragma
python run.py
```
This script runs a parametric study to study the evolution of the SIF with respect to the crack angle.

The script `display_results.py` shows a comparison between the analytical results and the FEM results.

## References

- Smith, D. J., Ayatollahi, M. R., & Pavier, M. J. (2001). The role of T-stress in brittle fracture for linear elastic materials under mixed-mode loading. Fatigue & Fracture of Engineering Materials & Structures, 24(2), 137–150. [https://doi.org/10.1046/j.1460-2695.2001.00377.x](https://doi.org/10.1046/j.1460-2695.2001.00377.x)
