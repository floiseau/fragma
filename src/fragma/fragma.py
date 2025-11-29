"""
fragma
======

This is the entry point of fragma.
This script parses the parameter file, generate the problem object, and run its solve method.
"""

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

    # Read the parameter file
    with open("parameters.toml", "rb") as toml_file:
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
