# Path-following using nodal displacement increment


## Description

This exemple simulates the crack propagation in a CT specimen with a load control method based nodal displacement increment.
The boundary conditions uses an force control.
This loading is controlled using by the displacement difference between two points, each on a face of the notch.
It correponds to a controlled of the loading based on the Crack Mouth Openning Displacement (CMOD).

## Expected results

In this example, we expect the CMOD to grow linearly (after the first 2 steps) with the load step number.
The slope should be equal to the imposed $\Delta \tau$.
A Python script, named `display_results.py`, is provided to display the evolution of the CMOD with the load steps and the force-CMOD curve.

