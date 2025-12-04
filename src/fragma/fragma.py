"""
fragma
======

This is the entry point of fragma.
This script parses the parameter file, generate the problem object, and run its solve method.
"""

import argparse
import tomllib

from .problems import ElasticityProblem, FractureProblem


def run_fragma():
    # Display header
    print(
        """
    ███████ ██████   █████   ██████  ███    ███  █████  
    ██      ██   ██ ██   ██ ██       ████  ████ ██   ██ 
    █████   ██████  ███████ ██   ███ ██ ████ ██ ███████ 
    ██      ██   ██ ██   ██ ██    ██ ██  ██  ██ ██   ██ 
    ██      ██   ██ ██   ██  ██████  ██      ██ ██   ██

    Fracture in Anisotropic Media using a Phase-field Model

    Author(s):
        Flavien Loiseau (flavien.loiseau@ensta-paris.fr)
    """
    )

    # Get the CLI arguments
    parser = argparse.ArgumentParser(
        description="Run fragma with a specified config file."
    )
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="Path to the TOML config file (default: parameters.toml)",
        default="parameters.toml",
    )
    args = parser.parse_args()

    # Read the parameter file
    with open(args.config_file, "rb") as toml_file:
        pars = tomllib.load(toml_file)

    # Choose the problem
    model = pars["model"]["name"]
    match model:
        case "elasticity":
            problem = ElasticityProblem(pars)
        case "fracture":
            problem = FractureProblem(pars)

    # Run the solver
    problem.solve()


if __name__ == "__main__":
    run_fragma()
