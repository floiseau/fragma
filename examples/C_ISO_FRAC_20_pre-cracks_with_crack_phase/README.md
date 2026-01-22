# Pre-cracks with crack phase

## Description

This example introduces pre-cracking with the crack field on a Double-Edge-Notched Tensile (DENT) specimen.
Instead of representing the crack explicitely in the mesh, it is embedded in the initial crack field.
To use this feature, the section `initial_crack` must be filled in the parameter file.
```toml
[[initial_crack]]
p1 = [0, 0.5, 0]
p2 = [0.25, 0.5, 0]
width = 0.01

[[initial_crack]]
p1 = [0.75, 0.5, 0]
p2 = [1, 0.5, 0]
width = 0.01
```
Here, two pre-cracks are specified:
- the left crack going from the point $(0, 0.5, 0)$ to the point $(0.25, 0.5, 0)$ with a width of 0.01,
- the right crack going from the point $(0, 0.5, 0)$ to the point $(0.25, 0.5, 0)$ with a width of 0.01.
In practice, the width should be equal to the element size in the crack region.

*Remark:* The double brackets means that we make a array of table/dict (see [toml reference](https://toml.io/en/v1.0.0#array-of-tables)).

## Expected results

In the results, we can observe that the initial crack field contains both cracks.
