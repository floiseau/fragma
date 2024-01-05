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
    Solver for 2D elasticity problem (in plane strain or plain stress).
    The loading are proportional to time.
    """

    def __init__(self, pars):
        ### Parameters
        print("\n████ PARAMETERS")
        # Store paramters
        self.pars = pars
        # Display a summary
        print(json.dumps(self.pars, indent=4))
        # Define the domain
        self.domain, cell_tags, facet_tags = self.define_domain()
        # Define the state variables
        self.define_state_variables()
        # Define the boundary condition functions
        self.define_boundary_condition_functions(facet_tags)
        # Define the energy
        self.define_total_energy()
        # Define the displacement problem
        self.define_displacement_problem()

    def define_domain(self):
        print("\n████ DEFINITION OF THE DOMAIN")
        # Get the dimension
        dim = self.pars["model"]["dim"]
        # Read the mesh from GMSH
        msh_file = self.pars["mesh"]["msh_file"]
        print("Mesh reading output:")
        return io.gmshio.read_from_msh(msh_file, MPI.COMM_WORLD, gdim=dim)

    def define_state_variables(self):
        ### Variational formulation
        print("\n████ DEFINITION OF THE STATE VARIABLES")
        # Define the elements
        element_u = ufl.VectorElement("Lagrange", self.domain.ufl_cell(), 1)
        # Define finite element spaces
        self.V_u = fem.FunctionSpace(self.domain, element_u)
        # Define the state variables
        u = fem.Function(self.V_u, name="Displacement")
        # Define the state vector
        self.state = {"u": u}

    def eps(self, u):
        return ufl.sym(ufl.grad(u))

    def sig(self, u):
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
        return la * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0 * mu * self.eps(u)

    def define_total_energy(self):
        # Get the integrands
        dx = ufl.Measure("dx", domain=self.domain)
        ds = ufl.Measure("ds", domain=self.domain)
        # Define the imposed stress on the remaining of the boundary
        T = fem.Constant(self.domain, default_scalar_type((0, 0)))
        # Define the volumic forces
        f = fem.Constant(self.domain, default_scalar_type((0, 0)))
        # Get state variables
        u = self.state["u"]
        # Define the energy terms
        elastic_energy = 0.5 * ufl.inner(self.sig(u), self.eps(u)) * dx
        dissipated_energy = 0.0 * dx
        external_work = ufl.dot(f, u) * dx + ufl.dot(T, u) * ds
        # Define the total energy
        self.total_energy = elastic_energy + dissipated_energy - external_work

    def define_boundary_condition_functions(self, facet_tags):
        ### Locate Boundary
        print("\n████ LOCATE BOUNDARIES")
        # Get the physical groups (mapping between pg and their indices)
        facets_tags_values = self.pars["mesh"]["physical_groups"]
        # Get the facets indices
        boundary_facets = {}
        for facet_name, facet_value in facets_tags_values.items():
            boundary_facets[facet_name] = facet_tags.indices[
                facet_tags.values == facet_value
            ]
        # Get the dimensions of domain and facets
        dim = self.domain.topology.dim
        fdim = self.domain.topology.dim - 1
        # Get boundary dofs (per comp)
        boundaries = {
            f"{facet_name}_{comp}": fem.locate_dofs_topological(
                (self.V_u.sub(comp), self.V_u.sub(comp).collapse()[0]),
                fdim,
                boundary_facet,
            )
            for comp in range(dim)
            for facet_name, boundary_facet in boundary_facets.items()
        }

        print("\n████ INITIALIZE BOUNDARY CONDITIONS")
        # Get displacement increments
        u_incs = self.pars["loading"]["u_incs"]
        # Create variables to store bcs and loading functions
        self.bcs = []
        self.load_funcs = {}
        # Iterage through the displacement increments
        for facet_name, u_inc in u_incs.items():
            # Get the component number
            comp = int(facet_name.split("_")[-1])
            # Define an FEM function (to control the BC)
            self.load_funcs[facet_name] = fem.Function(self.V_u.sub(comp).collapse()[0])
            # Update the load
            with self.load_funcs[facet_name].vector.localForm() as bc_local:
                bc_local.set(u_inc)
            # Add the boundary conditions to the list
            self.bcs.append(
                fem.dirichletbc(
                    self.load_funcs[facet_name], boundaries[facet_name], self.V_u
                )
            )

    def define_displacement_problem(self):
        print("\n████ DEFINITION OF THE DISPLACEMENT PROBLEM")
        # Get the state variables
        u = self.state["u"]
        # Derivative of the energy with respect to displacement to obtain the linear problem to determine the stationary point
        E_u = ufl.derivative(self.total_energy, u, ufl.TestFunction(self.V_u))
        E_du = ufl.replace(E_u, {u: ufl.TrialFunction(self.V_u)})
        # Define the displacement problem
        self.problem_u = LinearProblem(
            a=ufl.lhs(E_du),
            L=ufl.rhs(E_du),
            bcs=self.bcs,
            u=u,
            petsc_options={"ksp_type": "preonly", "pc_type": "cholesky"},
        )

    def update_boundary_conditions(self, t: float):
        # Get displacement increments
        u_incs = self.pars["loading"]["u_incs"]
        # Iterate through the load functions
        for facet_name, load_func in self.load_funcs.items():
            # Increment the load function
            with load_func.vector.localForm() as bc_local:
                bc_local.set(default_scalar_type(t * u_incs[facet_name]))

    def solve(self):
        print("\n████ RESOLUTION")
        # Start export
        self.init_export()
        # Start the loading iterations
        t_max = self.pars["loading"]["t_max"]
        for t in range(t_max + 1):
            # Display information
            print(f"== Load step {t}/{t_max}")
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
