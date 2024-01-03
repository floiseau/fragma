from pathlib import Path

from mpi4py import MPI
import numpy as np

from dolfinx import io
from dolfinx.io.gmshio import read_from_msh
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

class ElasticitySolver():
    """
    TODO
    """

    def __init__(self):
        ### Parameters
        print("=== Parameters")
        # Geometry
        self.dim = 2
        self.msh_file = "mesh.msh"
        # Elasticity
        self.pars = {"mu": 10e9, "lambda": 20e9}
        # Boundary conditions

    def setup_problem(self):
        ### Domain
        print("=== Domain")
        # Read the mesh from GMSH
        self.domain, cell_tags, facet_tags = io.gmshio.read_from_msh(
                self.msh_file,
                MPI.COMM_WORLD,
                gdim=self.dim)
        # Define the elements
        element = ufl.VectorElement("Lagrange", self.domain.ufl_cell(), 1)
        # Define finite element spaces
        V = fem.FunctionSpace(self.domain, element)
        # Get the boundary integrand
        ds = ufl.Measure("ds", domain=self.domain)

        ### Boundary Conditions
        print("=== Boundary conditions")
        # Define the map between facet names and facet values
        facets_tags_values = {"left" : 5,
                              "right": 6}
        # Define the imposed displacements
        bcs_u = {"left" : np.array([0, 0],       dtype=default_scalar_type),
                 "right": np.array([1e-3, 1e-3], dtype=default_scalar_type)}
        # Get the facets indices
        boundary_facets = {}
        for facet_name, facet_value in facets_tags_values.items():
            boundary_facets[facet_name] = facet_tags.indices[facet_tags.values == facet_value]
        # Get the dimension of facets
        fdim = self.domain.topology.dim - 1
        # Get boundary dofs
        boundaries = {facet_name:
                      fem.locate_dofs_topological(V, fdim, boundary_facet)
                      for facet_name, boundary_facet in boundary_facets.items()}
        # Define boundary conditions
        self.bcs = [fem.dirichletbc(bcs_u[facet_name], boundary, V)
                    for facet_name, boundary in boundaries.items()]
        # Define the imposed stress on the remaining of the boundary
        T = fem.Constant(self.domain, default_scalar_type((0, 0)))
        # Define the volumic forces
        f = fem.Constant(self.domain, default_scalar_type((0, 0)))

        ### Variational formulation
        print("=== Variational formulation")
        # Define strain
        def epsilon(u):
            return ufl.sym(ufl.grad(u))
        # Define stress
        def sigma(u):
            mu, la = self.pars["mu"], self.pars["lambda"]
            return la*ufl.nabla_div(u)*ufl.Identity(len(u)) + 2.*mu*epsilon(u)
        # Define the unknonw fields
        u = ufl.TrialFunction(V)
        # Define the test fields
        v = ufl.TestFunction(V)
        # Define the problem
        self.a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        self.L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

        ### Assembly of the problem
        # Define the linear problem
        self.elasticity_problem = LinearProblem(
                self.a, self.L, bcs=self.bcs,
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
                )

    def solve(self):
        print("=== Resolution")
        self.uh = self.elasticity_problem.solve()
    
    def export(self):
        ### Export
        print("=== Export")
        # Setup export
        results_folder = Path("results")
        results_folder.mkdir(exist_ok=True, parents=True)
        filename = results_folder / "fundamentals"
        # XDMF export
        with io.XDMFFile(self.domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
            xdmf.write_mesh(self.domain)
            self.uh.name = "u"
            xdmf.write_function(self.uh)
