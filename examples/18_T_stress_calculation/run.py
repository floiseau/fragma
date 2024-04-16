#!/usr/bin/env python

import itertools
import os
import subprocess
from multiprocessing import Pool

import numpy as np


def run_simulation(Ri_Re, N):
    """Function to perform the simulation for a given set of parameters."""
    # Get the radii
    Ri, Re = Ri_Re
    # Create directory for simulation
    dir_name = f"{N=:05d}_{Ri=:0.3g}_{Re=:0.3g}"
    os.makedirs(dir_name, exist_ok=True)

    # Execute gmsh command to generate mesh
    gmsh_command = [
        "gmsh",
        "-2",
        "base/mesh.geo",
        "-setnumber",
        "N",
        str(N),
        "-o",
        f"{dir_name}/mesh.msh",
    ]
    with open(f"{dir_name}/gmsh.log", "w") as log_file:
        subprocess.run(gmsh_command, stdout=log_file, stderr=log_file, text=True)

    # Copy parameters file
    with open("base/parameters.toml", "r") as par_file:
        content = par_file.read()
    content = content.format(R_int=Ri, R_ext=Re)
    with open(f"{dir_name}/parameters.toml", "w") as par_file:
        par_file.write(content)
    # Change to the directory and run the simulation
    os.chdir(dir_name)

    # Set the env variables
    my_env = os.environ.copy()
    my_env["OMP_NUM_THREADS"] = "1"
    # Run the simulation using Fragma
    fragma_command = [
        "python",
        "/home/flavien.loiseau/ownCloud/codes/fragma/fragma/main.py",
    ]
    with open("fragma.log", "w") as log_file:
        res = subprocess.run(
            fragma_command, stdout=log_file, stderr=log_file, text=True, env=my_env
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
    a = 0.05
    Ri_Re_list = [(a / 4, a / 2), (a / 8, a / 4), (a / 16, a / 8), (a / 32, a / 16)]
    N_list = [32, 64, 128]

    # Generate the combinations of parameters
    args = list(reversed(list(itertools.product(Ri_Re_list, N_list))))

    # Run the simulation in parallel
    with Pool(min(len(args), os.cpu_count() - 4)) as p:
        p.starmap(run_simulation, args)
