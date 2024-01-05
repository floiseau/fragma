from models.elasticity import Elasticity2DSolver

# TODO
#   Loading
#       Impose the displacement in the "volume"
#   Elasticity
#       Compute and export the strain
#       Compute and export the stress
#   Cleaning
#       Might be able to separate the definition of the energy from the other of the code
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

# Use the solver
solver = Elasticity2DSolver()
solver.solve()
