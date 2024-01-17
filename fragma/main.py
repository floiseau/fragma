import tomllib

from dolfinx import log

from problems.elasticity import ElasticityProblem
from problems.fracture import FractureProblem

# TODO
#   Cleaning
#       Transform the current solvers into problems
#           Problem should manage FEM space, state variables and
#           Ideally, it should also contain PETScProblem that PETSc can take as input
#       Create solver classes taking a problem as input and solving it using PETSc interface if possible
#   Fracture
#       Implement the staggered path-following approach (based on dissipation or the integral of crack phase)
#   Post-process
#       Compute and export the strain
#       Compute and export the stress

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

# # Set the log level
# log.set_log_level(log.LogLevel.INFO)

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
    case "fracture_monolithic":
        # solver = FractureMonolithicSolver(pars)
        raise NotImplementedError(f"Model '{model}' is not implemented.")
    case "fracture_path_following":
        # solver = FracturePathFollowingSolver(pars)
        raise NotImplementedError(f"Model '{model}' is not implemented.")
    case _:
        raise NotImplementedError(f"Model '{model}' is not implemented.")

# Run the solver
problem.solve()
