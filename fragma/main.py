from models.elasticity import ElasticitySolver

# TODO
#   Cleaning
#       Make a recap of the parameters (in initialization)
#       Add a separate method to update the loading (or set the loading)
#   Elasticity
#       Add plane stress
#   Time-dependent
#       Time-dependent problem https://jsdokken.com/dolfinx-tutorial/chapter2/diffusion_code.html
#   Fracture
#       Add a residual stiffness
#       Crack phase=1 at crack tip? Or along the whole crack lips?

# Display information
print("""
███████ ██████   █████   ██████  ███    ███  █████  
██      ██   ██ ██   ██ ██       ████  ████ ██   ██ 
█████   ██████  ███████ ██   ███ ██ ████ ██ ███████ 
██      ██   ██ ██   ██ ██    ██ ██  ██  ██ ██   ██ 
██      ██   ██ ██   ██  ██████  ██      ██ ██   ██
""")
print("Fracture in Anisotropic Media using a Phase-field Model")
print("")
print("Author(s):")
print("    Flavien Loiseau (flavien.loiseau@ensta-paris.fr)")
print("")

# Use the solver
solver = ElasticitySolver()
solver.setup_problem()
solver.solve()
solver.export()
