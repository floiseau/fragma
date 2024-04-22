# Calculation of the energy release rate with $G-\theta$ method

## Description
This example is dedicated to the post-process of the stress intensity factors in 2D.
The calculation are based on the $I$-integral using a domain integral (using $G(\theta)$ method).

Here a Center-Crack Tensile (CCT) specimen is considered.
To obtain the stress intensity factors under a loading, the `postprocess` section of the paramter file must be filled as shown below
```toml
[postprocess.SIFs]
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
According to Smith *et al.* (2001), the $T$-stress can be expressed as
```math

K_{I} = \sigma \sqrt(\pi a) \cos(2*\alpha),
\quad
K_{II} = \sigma \sqrt(\pi a) \cos(\alpha) \sin(\alpha),
```
where $\sigma$ is the applied stress along $y$-axis, $a$ is the crack half length, $\alpha$ is the initial crack angle.

In our case, we have
```math
\sigma = 1
, \quad
\alpha = 0
, \quad
a = 0.05,
```
where all quantities are expressed in SI unit.

We obtain $K_{I} = 0.396$ Pa$\cdot\sqrt{\mathrm{m}}$ and $K_{II} = 0.0$ Pa$\cdot\sqrt{\mathrm{m}}$.

## Running the example

In order to run this exemple, the following commands should be executed.
```shell
conda activate fragma
python run.py
```
This script runs a parametric study on both the mesh size $h$ and the interior radius $R_{\mathrm{int}}$.
In practice, the parameters are:
- $R_{\mathrm{int}}$ the interior radius, the exterior radius $R_{\mathrm{ext}}$ is chosen as $R_{\mathrm{ext}} = R_{\mathrm{int}} / 2$,
- $N$ the mesh size parameter, the mesh size at boundary is $h = L/N$ and the mesh size at crack tip is $h_{\mathrm{crack}} = h/100 = L / (100 N)$.

The script `display_results.py` shows a scatter plot with the evolution of $K_{I}$ has a function of both parameters.

## Expected results

In this exemple, the `probes.csv` files contains the columns $K_{I}$ and $K_{II}$.
Running the example gives $K_{I} \approx 0.399$ Pa$\cdot\sqrt{\mathrm{m}}$ which is close the analytical solution.

## References

- Smith, D. J., Ayatollahi, M. R., & Pavier, M. J. (2001). The role of T-stress in brittle fracture for linear elastic materials under mixed-mode loading. Fatigue & Fracture of Engineering Materials & Structures, 24(2), 137–150. [https://doi.org/10.1046/j.1460-2695.2001.00377.x](https://doi.org/10.1046/j.1460-2695.2001.00377.x)
