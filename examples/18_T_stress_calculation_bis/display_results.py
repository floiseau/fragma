import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Parameters
sig = 1
a = 0.05
alphas_deg = np.array(sorted([alpha for alpha in range(0, 90 + 1, 5)]))
alphas = np.deg2rad(alphas_deg)


# Generate the combination of parameters
N = len(alphas)

# Read the csv files
G_res = np.empty((N,))
K1_res = np.empty((N,))
K2_res = np.empty((N,))
T_res = np.empty((N,))
Beq_res = np.empty((N,))
for i, alpha_deg in enumerate(alphas_deg):
    # Set the dir name
    dir_name = f"{alpha_deg=:02d}"
    # Read the result file
    df = pd.read_csv(f"{dir_name}/results/probes.csv")
    # Get the SIFs
    G_res[i] = df["G"][1]
    K1_res[i] = df["K_I"][1] / (sig * np.sqrt(np.pi * a))
    K2_res[i] = df["K_II"][1] / (sig * np.sqrt(np.pi * a))
    T_res[i] = df["T_stress"][1] / sig
    Beq_res[i] = (
        df["T_stress"][1]
        * np.sqrt(np.pi * a)
        / np.sqrt(df["K_I"][1] ** 2 + df["K_II"][1] ** 2)
    )

# Define the analytical solution
# beta = np.pi / 2 - alphas # Why does it not match with Smith et al. (2001)? Did I made a mistake?
# K1_ana = np.sin(2 * beta)
# K2_ana = np.cos(beta) * np.sin(beta)
# T_ana = np.cos(2 * beta)
K1_ana = np.cos(alphas) ** 2
K2_ana = np.cos(alphas) * np.sin(alphas)
T_ana = -np.cos(2 * alphas)
Beq_ana = np.cos(alphas) * (np.tan(alphas) ** 2 - 1)

# Define the literature solution by Yu and Kuna (2021)
alphas_deg_YuK = np.array([0, 15, 30, 45, 60, 75, 90])
K1_YuK = np.array([1.014, 0.946, 0.760, 0.506, 0.253, 0.068, 0.000])
K2_YuK = np.array([0.000, 0.253, 0.437, 0.504, 0.437, 0.252, 0.000])
T_YuK = np.array([-1.022, -0.886, -0.515, -0.009, 0.496, 0.865, 1.000])

# Display the results
fig = plt.figure()
plt.plot(alphas_deg, K1_ana, c="#003162", label="Ana - Smith et al. (2001)")
plt.scatter(alphas_deg_YuK, K1_YuK, c="#007e9f", label="FEM - Yu and Kuna (2021)")
plt.scatter(alphas_deg, K1_res, c="#f1443a", label="FEM - Us")
plt.xlabel("alpha $\\alpha$ (°)")
plt.ylabel("$\\frac{K_{\\mathrm{I}}}{\\sigma \\sqrt{\\pi a}}$")
plt.grid()
plt.legend()

fig = plt.figure()
plt.plot(alphas_deg, K2_ana, c="#003162", label="Smith et al. (2001)")
plt.scatter(alphas_deg_YuK, K2_YuK, c="#007e9f", label="FEM - Yu and Kuna (2021)")
plt.scatter(alphas_deg, K2_res, c="#f1443a", label="FEM")
plt.xlabel("alpha $\\alpha$ (°)")
plt.ylabel("$\\frac{K_{\\mathrm{II}}}{\\sigma \\sqrt{\\pi a}}$")
plt.grid()
plt.legend()

fig = plt.figure()
plt.plot(alphas_deg, T_ana, c="#003162", label="Smith et al. (2001)")
plt.scatter(alphas_deg_YuK, T_YuK, c="#007e9f", label="FEM - Yu and Kuna (2021)")
plt.scatter(alphas_deg, T_res, c="#f1443a", label="FEM")
plt.xlabel("alpha $\\alpha$ (°)")
plt.ylabel("$T$-stress (unit TODO)")
plt.grid()
plt.legend()


fig = plt.figure()
plt.plot(alphas_deg[:-1], Beq_ana[:-1], c="#003162", label="Smith et al. (2001)")
plt.scatter(alphas_deg[:-1], Beq_res[:-1], c="#f1443a", label="FEM")
plt.xlabel("alpha $\\alpha$ (°)")
plt.ylabel("$B_{\\mathrm{eq}}$")
plt.grid()
plt.legend()


plt.show()
