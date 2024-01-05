import tomllib
import json
from pathlib import Path
import pprint

from mpi4py import MPI
import numpy as np

from dolfinx import io
from dolfinx.io.gmshio import read_from_msh
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl


class Elasticity2DSolver:
    """
    TODO
    """

    def __init__(self):
        ### Parameters
        print("\n████ PARAMETERS")
        # Read the parameter file
        with open("parameters.toml", "rb") as toml_file:
            self.pars = tomllib.load(toml_file)
        # Load some main parameters
        self.dim = self.pars["model"]["dim"]
        # Boundary conditions
        self.facets_tags_values = self.pars["mesh"]["physical_groups"]
        # Get the loading parameters
        self.t_max = self.pars["loading"]["t_max"]
        self.u_incs = self.pars["loading"]["u_incs"]
        # Display a summary
        print(json.dumps(self.pars, indent=4))
        # Define the problem
        self.define_problem()

    def define_problem(self):
        ### Domain
        print("\n████ DOMAIN")
        # Read the mesh from GMSH
        msh_file = self.pars["mesh"]["msh_file"]
        print("Mesh reading output:")
        self.domain, cell_tags, facet_tags = io.gmshio.read_from_msh(
            msh_file, MPI.COMM_WORLD, gdim=self.dim
        )
        # Define the elements
        element_u = ufl.VectorElement("Lagrange", self.domain.ufl_cell(), 1)
        # Define finite element spaces
        V_u = fem.FunctionSpace(self.domain, element_u)
        ### Locate Boundary
        print("\n████ LOCATE BOUNDARIES")
        # Read the mesh from GMSH
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
        ### Apply loads
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
        print("\n████ VARIATIONAL FORMULATION")
        # Define the state variables
        u = fem.Function(V_u, name="Displacement")
        # Define the state vector
        self.state = {"u": u}

        # Define strain
        def eps(u):
            return ufl.sym(ufl.grad(u))

        # Define stress
        def sig(u):
            # Get the elastic parameters
            E = self.pars["mechanical"]["E"]
            nu = self.pars["mechanical"]["nu"]
            # Compute Lame coefficient
            la = E * nu / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
            # Check the 2D assumption
            assumption = self.pars["model"]["2D_assumption"]
            match assumption:
                case "plane_stress":
                    print("Plane stress assumption")
                    la = 2 * mu * la / (la + 2 * mu)
                case "plane_strain":
                    print("Plane strain assumption")
                case _:
                    raise ValueError(f'The 2D assumption "{assumption}" in unknown')
            # Compute the stess
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
            petsc_options={"ksp_type": "preonly", "pc_type": "cholesky"},
        )

    def update_boundary_conditions(self, t: float):
        # Iterate through the load functions
        for facet_name, load_func in self.load_funcs.items():
            # Increment the load function
            with load_func.vector.localForm() as bc_local:
                bc_local.set(default_scalar_type(t * self.u_incs[facet_name]))

    def solve(self):
        print("\n████ RESOLUTION")
        # Start export
        self.init_export()
        # Start the loading iterations
        for t in range(self.t_max + 1):
            # Display information
            print(f"== Load step {t}/{self.t_max}")
            # Update boundary conditions
            self.update_boundary_conditions(t)
            # Solve the displacement problem
            self.problem_u.solve()
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
        self.xdmf_file.write_function(self.state["u"], t)

    def end_export(self):
        # Close the file
        self.xdmf_file.close()
