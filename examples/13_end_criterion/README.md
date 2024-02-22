# End criterion

This example is dedicated to the introduction of the end criteria.
The end criterion is set in the `parameters.toml` file in the `loading` section.

```toml
[loading]
end_criterion = "elastic_energy_drop"
```
Its value can be:
- `"t"`, then the simulation ends when the time reaches 1,
- `"elastic_energy_drop"`, then the simulation ends when the elastic energy reaches 1% of its maximum value during the simulation.

