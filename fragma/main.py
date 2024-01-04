from models.elasticity import ElasticitySolver

# TODO
#   Cleaning
#       Make a recap of the parameters (in initialization)
#       Might be able to separate the definition of the energy from the other of the code
#   Elasticity
#       Add plane stress
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
solver = ElasticitySolver()
solver.solve()
