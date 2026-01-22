# Heterogeneous material properties

## Description
This example is dedicated to the introduction of heterogeneous material properties, in particular the critical energy release rate $G_c$.
To use this feature, the expression of $G_c$ as a function of the coordinate `x[0], x[1]` (and `x[2]` in 3D) must be given as a string.
In this example, it is defined as,
```math
G_c(x,y) = 380 \times (1 + 2xy).
```
It is written as follows in the paramter file.
```toml
Gc = "380*(1+2*x[0]*x[1])"
```
In practice, this expression is read by sympy using the lambdify function.
Thus, it can use any mathetical function available in sympy.

## Results
In this example, the crack must be deviated toward the bottom due to $G_c$ increasing toward the upper-right corner.
Increasing the factor in front of $xy$ should further deviate the crack.

