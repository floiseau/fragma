# Path-following using the undamaged elastic energy


## Description

This exemple simulates the crack propagation in a CT specimen with a load control method based the undamaged elastic energy.
Note that the boundary conditions uses an force control.

## Expected results

In this example, we expect the square root of the undamaged energy to grow linearly (after the first 2 steps) with a slope equal to the imposed $\Delta \tau$.
A Python script, named `display_results.py`, is provided to display the force/displacement curve and the evolution of the square root of the undamaged elastic energy.

