# Path-following using max in time of the crack driving variable

## Description

This example simulates the crack propagation in a CT specimen with a load control method based the maximum in time of the crack driving variable.
The boundary conditions use a force control.

Using this method is not recommended as its performances are highly sensitive to the choice of $\dtau$.
If the step size $\dtau$ is too small, the load step might stagnate, whereas when the step size $\dtau$ is too large, the load step might too large and full failure will occurs in one load step.

## Expected results

A Python script, named `display_results.py`, is provided to display the evolution of the CMOD with the load steps and the force-CMOD curve.
