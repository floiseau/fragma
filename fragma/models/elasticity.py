from pathlib import Path

from mpi4py import MPI
import numpy as np

from dolfinx import io
from dolfinx.io.gmshio import read_from_msh
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl


class ElasticitySolver:
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

        self.facets_tags_values = {
            "left": 5,  # map between facet names and facet values
            "right": 6,
        }
        # Define the displacement increments
        self.u_incs = {
            "left_0": default_scalar_type(0),
            "left_1": default_scalar_type(0),
            "right_0": default_scalar_type(0),
            "right_1": default_scalar_type(1e-3),
        }

        # Define the problem
        self.define_problem()

    def define_problem(self):
        ### Domain
        print("=== Domain")
        # Read the mesh from GMSH
        self.domain, cell_tags, facet_tags = io.gmshio.read_from_msh(
            self.msh_file, MPI.COMM_WORLD, gdim=self.dim
        )
        # Define the elements
        element_u = ufl.VectorElement("Lagrange", self.domain.ufl_cell(), 1)
        # Define finite element spaces
        V_u = fem.FunctionSpace(self.domain, element_u)

        ### Boundary Conditions
        print("=== Boundary conditions")
        # Get the facets indices
        boundary_facets = {}
        for facet_name, facet_value in self.facets_tags_values.items():
            boundary_facets[facet_name] = facet_tags.indices[
                facet_tags.values == facet_value
            ]
        # Get the dimension of facets
        fdim = self.domain.topology.dim - 1
        # Get boundary dofs (per comp)
        boundaries = {
            f"{facet_name}_{comp}": fem.locate_dofs_topological(
                (V_u.sub(comp), V_u.sub(comp).collapse()[0]), fdim, boundary_facet
            )
            for comp in range(self.dim)
            for facet_name, boundary_facet in boundary_facets.items()
        }
        # Varying boundary conditions
        bcs = []
        self.load_funcs = {}
        for facet_name, u_inc in self.u_incs.items():
            # Get the component number
            comp = int(facet_name.split("_")[-1])
            # Define an FEM function (to control the BC)
            self.load_funcs[facet_name] = fem.Function(V_u.sub(comp).collapse()[0])
            # Update the load
            with self.load_funcs[facet_name].vector.localForm() as bc_local:
                bc_local.set(u_inc)
            # Add the boundary conditions to the list
            bcs.append(
                fem.dirichletbc(
                    self.load_funcs[facet_name], boundaries[facet_name], V_u
                )
            )
        # Define the imposed stress on the remaining of the boundary
        T = fem.Constant(self.domain, default_scalar_type((0, 0)))
        # Define the volumic forces
        f = fem.Constant(self.domain, default_scalar_type((0, 0)))

        ### Variational formulation
        print("=== Variational formulation")
        # Define the state variables
        u = fem.Function(V_u, name="Displacement")

        # Define strain
        def eps(u):
            return ufl.sym(ufl.grad(u))

        # Define stress
        def sig(u):
            mu, la = self.pars["mu"], self.pars["lambda"]
            return la * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0 * mu * eps(u)

        # Get the integrands
        dx = ufl.Measure("dx", domain=self.domain)
        ds = ufl.Measure("ds", domain=self.domain)
        # Define the energy
        elastic_energy = 0.5 * ufl.inner(sig(u), eps(u)) * dx
        external_work = ufl.dot(f, u) * dx + ufl.dot(T, u) * ds
        total_energy = elastic_energy - external_work

        # Derivative of the energy
        E_u = ufl.derivative(total_energy, u, ufl.TestFunction(V_u))
        E_du = ufl.replace(E_u, {u: ufl.TrialFunction(V_u)})
        # Define the displacement problem
        self.problem_u = LinearProblem(
            a=ufl.lhs(E_du),
            L=ufl.rhs(E_du),
            bcs=bcs,
            u=u,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

    def update_boundary_conditions(self, t: float):
        # Iterate through the load functions
        for facet_name, load_func in self.load_funcs.items():
            # Increment the load function
            with load_func.vector.localForm() as bc_local:
                bc_local.set(t * self.u_incs[facet_name])

    def solve(self):
        print("=== Resolution")
        # Start export
        self.init_export()
        # Start the loading iterations
        for t in range(10):
            # Update boundary conditions
            self.update_boundary_conditions(t)
            # Solve the displacement problem
            self.uh = self.problem_u.solve()
            # Export the results
            self.export_state(t)
        # End export
        self.end_export()

    def init_export(self):
        # Create the export directory
        results_folder = Path("results")
        results_folder.mkdir(exist_ok=True, parents=True)
        # Set the name of the exported file
        filename = results_folder / "results"
        # Open the file
        self.xdmf_file = io.XDMFFile(
            self.domain.comm, filename.with_suffix(".xdmf"), "w"
        )
        # Export the mesh
        self.xdmf_file.write_mesh(self.domain)

    def export_state(self, t):
        # Export displacement file
        self.xdmf_file.write_function(self.uh, t)

    def end_export(self):
        # Close the file
        self.xdmf_file.close()
