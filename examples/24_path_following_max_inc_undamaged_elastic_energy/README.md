# Path-following using nodal displacement increment


## Description

This exemple simulates the crack propagation in a SENT test with a load control method based on the energy release with a regularization,
$$
\mathcal{D} + \beta \Delta \mathcal{U} - \Delta \tau = 0,
$$
where $\mathcal{D}$ is the energy dissipated during the loading increment, and $\Delta \mathcal{U}$ is the variation of elastic energy.
The parameter $\beta$ is a regularization parameter.
It gives a weighting between a control in elastic energy variation and a control in dissipation.
Note that the variation of elastic energy might be negative during the loading; thus, $\beta$ must be small.
However, if $\beta$ is too small, a large variation of the elastic energy is required at the beginning of the loading and the snap-back might be skipped.

The boundary conditions are imposed in displacement.
The control equation is configured in the parameter file through the following lines.

```toml
[loading]
dl = 0.01
constraint = "energy_release"
dtau = 0.01
beta = 1e-4
```
