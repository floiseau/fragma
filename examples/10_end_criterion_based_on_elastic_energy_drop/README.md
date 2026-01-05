# End criterion

This example is dedicated to the introduction of the end criteria.
The end criterion is set in the `parameters.toml` file in the `end` section.

Currrently, to end check are available.
The first one is the time end checker.
```toml
[end]
end_criterion = "t"
t_max = 50
```
In this case, the simulation stops after $50$ increments.

The second one is the elastic energy drop checker.
```toml
[end]
end_criterion = "elastic_energy_drop"
drop = 0.01
```
With the elastic energy drop checker the simulation ends when the elastic energy reaches $\mathrm{drop}\,%$ of its maximum value during the simulation.

