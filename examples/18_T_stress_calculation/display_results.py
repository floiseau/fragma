import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Parameters
a = 0.05
Ri_Re_list = [(a/4, a/2), (a/8, a/4), (a/16, a/8), (a/32, a/16)]
N_list = [32, 64, 128]

# Generate the combination of parameters
pars = list(itertools.product(Ri_Re_list, N_list))
N = len(pars)

# Read the csv files
N_res = np.empty((N,))
Ri_res = np.empty((N,))
Re_res = np.empty((N,))
T_res = np.empty((N,))
for i, (Ri_Re, N) in enumerate(pars):
    # Get the radii
    Ri, Re = Ri_Re
    # Set the dir name
    dir_name = f"{N=:05d}_{Ri=:0.3g}_{Re=:0.3g}"
    # Read the result file
    df = pd.read_csv(f"{dir_name}/results/probes.csv")
    # Get the T stress
    T = df["T_stress"][1]
    # Display the T-stress value
    print(f"{dir_name} =>", "T-stress is", T)
    # Store the results
    Ri_res[i] = Ri
    Re_res[i] = Re
    N_res[i] = N
    T_res[i] = T

# Display the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    N_res, Ri_res, T_res,
    linewidth=0, antialiased=False)
ax.set_xlabel("$N$")
ax.set_ylabel("$R_i$")
ax.set_zlabel("$T$")
plt.show()

