import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
B = 0.01

# Read the probe file
df = pd.read_csv("results/probes.csv")

# Comupte intermediate quantities
u1_bot = df["displacement 1 [-0.01       0.0239375  0.       ]"]
u2_bot = df["displacement 2 [-0.01       0.0239375  0.       ]"]
u1_top = df["displacement 1 [-0.01       0.0240625  0.       ]"]
u2_top = df["displacement 2 [-0.01       0.0240625  0.       ]"]


u_extenso = np.sqrt((u1_top - u1_bot) ** 2 + (u2_top - u2_bot) ** 2)
f_bot = -df["F_2 (bot_pin)"] * B


# Display the evolution of extenso (clip) measure during the loading
# NOTE We expect this quantity to grow linearly
plt.figure()
plt.plot(u_extenso)
plt.xlabel("Load step")
plt.ylabel("Extenso displacement (CMOD)")
plt.grid()

# Display the force-displacement curve
plt.figure()
plt.plot(u_extenso, f_bot, marker="x", linewidth=1)
plt.xlabel("Extenso displacement (CMOD)")
plt.ylabel("-F_2 (bot pin)")
plt.grid()

# Show the figures
plt.show()
