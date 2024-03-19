# Export of stress and strain


## Description

In the example, the stress and strain fields are exported in VTK files.
To enable the stress and strain exports, the following options must be added to the parameters files (in the postprocess section).
```toml
[postprocess]
fields = ["strain", "stress"]
```

## Expected results

In this example, the stress and strain files should be exported in the `results` directory.
To read them, one must open `Stress.pvd` and `Strain.pvd` in Paraview (or an other visualization tool).
