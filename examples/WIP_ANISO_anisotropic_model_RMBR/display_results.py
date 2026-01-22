import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Parameters
B = 1e-3  # Thickness of the specimen

### Load numerical data
# Read data
df = pd.read_csv("results/probes.csv", sep=",")
# Store plotted (and exported) data
res = {
    "u": df["displacement 2 [0.    0.001 0.   ]"],
    "F": df["F_2 (top)"] * B,
    "fracture_dissipation": (df["fracture_dissipation"] - df["fracture_dissipation"][0])*B,
}

### Generate the plots

## Force displacement curve
plt.figure()
# Plot numerical data using
plt.plot(
    res["u"],
    res["F"],
    linewidth=1,
    marker="x",
)
plt.xlabel("Vertical displacement on top boundary $u_y(y=L)$")
plt.ylabel("Force on top boundary $F_y(y=L)$")
plt.grid()

## Fracture dissipation 
plt.figure()
plt.plot(
    res["fracture_dissipation"],
    linewidth=1,
    marker="x",
)
plt.xlabel("Load step")
plt.ylabel("Dissipated energy (J)")
plt.grid()

# Show the plots
plt.show()
