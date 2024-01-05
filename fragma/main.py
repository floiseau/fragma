import tomllib

from models.elasticity import Elasticity2DSolver

# TODO
#   Loading
#       Impose the displacement in the "volume"
#   Elasticity
#       Compute and export the strain
#       Compute and export the stress
#   Cleaning
#       Might be able to separate the definition of the energy from the other in the code
#   Fracture
#       Add a residual stiffness
#       Crack phase=1 at crack tip? Or along the whole crack lips?

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

# Choose the solver
model = pars["model"]["name"]
match model:
    case "elasticity":
        if pars["model"]["dim"] == 2:
            solver = Elasticity2DSolver(pars)
        else:
            raise NotImplementedError(f"Solver for 3D elasticity is not implemented.")
    case _:
        raise NotImplementedError(f"Model '{model}' is not implemented.")

# Run the solver
solver.solve()
