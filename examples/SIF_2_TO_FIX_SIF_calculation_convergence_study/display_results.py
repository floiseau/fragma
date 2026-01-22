import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Parameters
a = 0.05
Ri_Re_list = [(a / 4, a / 2), (a / 8, a / 4), (a / 16, a / 8), (a / 32, a / 16)]
N_list = [32, 64, 128]

# Generate the combination of parameters
pars = list(itertools.product(Ri_Re_list, N_list))
N = len(pars)

# Read the csv files
N_res = np.empty((N,))
Ri_res = np.empty((N,))
Re_res = np.empty((N,))
K_I_res = np.empty((N,))
K_II_res = np.empty((N,))
for i, (Ri_Re, N) in enumerate(pars):
    # Get the radii
    Ri, Re = Ri_Re
    # Set the dir name
    dir_name = f"{N=:05d}_{Ri=:0.3g}_{Re=:0.3g}"
    # Read the result file
    df = pd.read_csv(f"{dir_name}/results/probes.csv")
    # Get the T stress
    K_I = df["K_I"][1]
    K_II = df["K_II"][1]
    # Display the T-stress value
    print(f"{dir_name} => {K_I=} and {K_II=}.")
    # Store the results
    Ri_res[i] = Ri
    Re_res[i] = Re
    N_res[i] = N
    K_I_res[i] = K_I
    K_II_res[i] = K_II

# Display the results
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(N_res, Ri_res, K_I_res, linewidth=0, antialiased=False)
ax.set_xlabel("$N$")
ax.set_ylabel("$R_i$")
ax.set_zlabel("$K_{I}$")

# Display the results
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(N_res, Ri_res, K_II_res, linewidth=0, antialiased=False)
ax.set_xlabel("$N$")
ax.set_ylabel("$R_i$")
ax.set_zlabel("$K_{II}$")
plt.show()
