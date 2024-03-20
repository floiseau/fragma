# Pre-cracks with crack phase

## Description

This example introduces pre-cracking with the crack field on a Double-Edge-Notched Tensile (DENT) specimen.
Instead of representing the crack explicitely in the mesh, it is embedded in the initial crack field.
To use this feature, the section `initial_cracks` must be filled in the parameter file.
```toml
[initial_cracks]
left_crack = [[0, 0.5, 0], [0.25, 0.5, 0]]
right_crack = [[0.75, 0.5, 0], [1, 0.5, 0]]
```
Here, two pre-cracks are specified:
- the left crack going from the point $(0, 0.5, 0)$ to the point $(0.25, 0.5, 0)$,
- the right crack going from the point $(0, 0.5, 0)$ to the point $(0.25, 0.5, 0)$,


## Expected results

In the results, we can observe that the initial crack field contains both cracks.
