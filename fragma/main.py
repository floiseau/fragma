from datetime import date
from pathlib import Path

from mpi4py import MPI
import numpy as np

from dolfinx import io
from dolfinx.io.gmshio import read_from_msh
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

# TODO Add a small summary for each step
# TODO With fracture: residual stiffness
# TODO Crack phase=1 at crack tip ?

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

### Parameters
print("=== Parameters")
# Geometry
dim = 2
msh_file = "mesh.msh"
# Elasticity
mu = default_scalar_type(10e9)
la = default_scalar_type(20e9)
# TODO Make a recap of the parameters

### Domain
print("=== Domain")
# Read the mesh from GMSH
domain, cell_tags, facet_tags = io.gmshio.read_from_msh(
        msh_file,
        MPI.COMM_WORLD,
        gdim=dim)

# Define finite element spaces
element = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
V = fem.FunctionSpace(domain, element)

# Get the boundary integrand
ds = ufl.Measure("ds", domain=domain)

### Boundary Conditions
print("=== Boundary conditions")
# Define the values
u_0 = np.array([0, 0],       dtype=default_scalar_type)
u_D = np.array([1e-3, 1e-3], dtype=default_scalar_type)
# Set the boundary tags
# TODO Automatize with the boundary name (see internship)
LEFT  = 5
RIGHT = 6
# Get the facets indices
boundary_facets_left  = facet_tags.indices[facet_tags.values == LEFT]
boundary_facets_right = facet_tags.indices[facet_tags.values == RIGHT]
# Get the dimensions of the boundary
fdim = domain.topology.dim - 1
# Get boundary dofs
boundary_left  = fem.locate_dofs_topological(V, fdim, boundary_facets_left)
boundary_right = fem.locate_dofs_topological(V, fdim, boundary_facets_right)
# Define boundary conditions
bc_left  = fem.dirichletbc(u_0, boundary_left,  V)
bc_right = fem.dirichletbc(u_D, boundary_right, V)
# Gathers bcs
bcs = [bc_right, bc_left]
# Free boundary for the remaining
T = fem.Constant(domain, default_scalar_type((0, 0)))
# Volumic forces
f = fem.Constant(domain, default_scalar_type((0, 0)))

### Variational formulation
print("=== Variational formulation")
# Define strain
def epsilon(u):
    return ufl.sym(ufl.grad(u))
# Define stress
def sigma(u):
    return la*ufl.nabla_div(u)*ufl.Identity(len(u)) + 2.*mu*epsilon(u)

# Define the unknonw fields
u = ufl.TrialFunction(V)
# Define the test fields
v = ufl.TestFunction(V)
# Define the problem
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds
# TODO print a recap of the formulation ???

### Solver
print("=== Resolution")
problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
uh = problem.solve()

### Export
print("=== Export")
# Setup export
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "fundamentals"
# VTK export
with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
    vtx.write(0.0)
# XDMF export
with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "u"
    xdmf.write_function(uh)
