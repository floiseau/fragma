#!/usr/bin/env python

import os
import subprocess
from multiprocessing import Pool

import numpy as np


def run_simulation(alpha_deg):
    """Function to perform the simulation for a given set of parameters."""
    # Set the crack length
    a = 0.05
    # Converte the angle from deg to rad
    alpha_rad = np.deg2rad(alpha_deg)
    # Create directory for simulation
    dir_name = f"{alpha_deg=:02d}"
    os.makedirs(dir_name, exist_ok=True)

    # Execute gmsh command to generate mesh
    gmsh_command = [
        "gmsh",
        "-2",
        "base/mesh.geo",
        "-setnumber",
        "alpha",
        str(alpha_deg),
        "-o",
        f"{dir_name}/mesh.msh",
    ]
    with open(f"{dir_name}/gmsh.log", "w") as log_file:
        subprocess.run(gmsh_command, stdout=log_file, stderr=log_file, text=True)

    # Copy parameters file
    with open("base/parameters.toml", "r") as par_file:
        content = par_file.read()
    content = content.format(
        ax=a * np.cos(alpha_rad),
        ay=a * np.sin(alpha_rad),
        alpha=alpha_deg,
    )
    with open(f"{dir_name}/parameters.toml", "w") as par_file:
        par_file.write(content)
    # Change to the directory and run the simulation
    os.chdir(dir_name)

    # Set the env variables
    my_env = os.environ.copy()
    my_env["OMP_NUM_THREADS"] = "1"
    # Run the simulation using Fragma
    with open("fragma.log", "w") as log_file:
        res = subprocess.run(
            ["fragma"], stdout=log_file, stderr=log_file, text=True, env=my_env
        )
    # Check if the simulation worked
    match res.returncode:
        case 0:
            print(f"SUCCESS - {dir_name}")
        case 1:
            print(f"FAILURE - {dir_name}")
        case _:
            print(f"UNKNOWN CODE {res.returncode} - {dir_name}")

    os.chdir("..")


if __name__ == "__main__":
    # Set the parameters
    angles = [alpha for alpha in range(0, 90 + 1, 5)]

    # Run the simulation in parallel
    with Pool(min(len(angles), os.cpu_count() - 4)) as p:
        p.map(run_simulation, angles)
