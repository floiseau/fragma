import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
B = 0.01

# Read the probe file
df = pd.read_csv("results/probes.csv")

# Comupte intermediate quantities
u1_bot = df["displacement 1 [0, 0.017, 0]"]
u2_bot = df["displacement 2 [0, 0.017, 0]"]
u1_top = df["displacement 1 [0, 0.02115, 0]"]
u2_top = df["displacement 2 [0, 0.02115, 0]"]
u_extenso = np.sqrt((u1_top - u1_bot) ** 2 + (u2_top - u2_bot) ** 2)
f_bot = -df["F_2 (bot_pin)"] * B


# Display the square root of the undamaged elastic energy
plt.figure()
plt.plot(np.sqrt(df["undamaged_elastic_energy"]))
plt.xlabel("Load step")
plt.ylabel("Square root of the undamaged elastic energy")
plt.grid()

# Display the force-displacement curve
plt.figure()
plt.plot(u_extenso, f_bot, marker="x", linewidth=1)
plt.xlabel("Displacement (clip)")
plt.ylabel("-F_2 (bot pin)")
plt.grid()

# Show the figures
plt.show()
