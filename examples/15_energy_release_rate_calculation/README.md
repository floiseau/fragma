# Calculation of the energy release rate with $G-\theta$ method

## Description
This example is dedicated to the post-process of the energy release rate.
The calculation are based on the $G-\theta$ method.

Here a Single Edge Notched Tension (SENT) specimen is considered.
To obtain the energy release rate under a displacement loading, the `postprocess` section of the paramter file must be filled as shown below
```toml
V_in = ...
V_tr = ...
```
where `V_in` is the (generalized) volume in which the $\theta$ field is equal to 1 and `V_tr` is the transition (generalized) volume where $\theta$ goes from 1 to 0.

## Approximate solution
According to Liu *et al.* (2015), the fracture toughness can be approximated
```math
K_I = \sigma \sqrt{\pi a} \left[ 1.122 - 0.231 \left(\frac{a}{b} \right) + 10.55 \left(\frac{a}{b}\right)^2 - 21.71 \left(\frac{a}{b}\right)^3 + 30.382 \left(\frac{a}{b}\right)^4  \right],
```
where $\sigma$ is the applied stress, $a$ is the crack length and $b$ is the width of the specimen.

Using the Irwin formula, the energy release rate can be expressed as:
```math
G = \frac{K_I^2}{E'},
\qquad
with E'=
\begin{cases}
    E,                  & \text{in plane stress}, \\
    \frac{E}{1-\nu^2},  & \text{in plane strain}.
\end{cases}
```

In our case, we have
```math
b = 10^{-3}
, \quad
a = 0.25 \times 10 ^{-3}
, \quad
\sigma = 1
, \quad
E = 230.77 \times 10^9
, \quad
\nu = 0.43,
```
where all quantities are expressed in SI unit.

We obtain $K_I = 42.12 \times 10^6$ Pa.m $^{-\frac{1}{2}}$.
Using Irwin formula, we obtain the energy release rate $G = 6267$ J/m $^2$.

## Expected results

In this exemple, the `probes.csv` files contains the energy release rate $G$.
Running the exemple gives a value around  $G=6269$ J/m $^2$ which is close the approximate solution given by Lui *et al.* (2015).

## References

- Liu, M., Gan, Y., Hanaor, D. A., Liu, B., & Chen, C. (2015). An improved semi-analytical solution for stress at round-tip notches. *Engineering fracture mechanics, 149*, 134-143.
