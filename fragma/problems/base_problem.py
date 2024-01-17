import json

from dolfinx import fem, default_scalar_type

from domain import Domain
from exporter import (
    VTKExporter,
    XDMFExporter,
    VTXExporter,
)  # TODO Remove unused imports
from postprocess import PostProcessor


class BaseProblem:
    def __init__(self, pars):
        ### Parameters
        print("\n████ PARAMETERS")
        # Store paramters
        self.pars = pars
        # Display a summary
        print(json.dumps(self.pars, indent=4))
        # Define the domain
        self.domain = Domain(pars["mesh"], pars["model"]["dim"])
        # Define the state variables
        self.define_state_variables()
        # Define problems
        self.define_problems()
        # Initialize post-processing
        self.postprocessor = PostProcessor(self.domain, self.model, self.state)
        # Initialize the exporter
        functions_to_export = list(self.state.values()) + list(
            self.postprocessor.funcs.values()
        )
        self.exporter = VTKExporter(self.domain.mesh, functions_to_export)

    def define_state_variables(self):
        raise NotImplementedError(
            "Solver: The method 'define_state_variables' must be implemented in the child class."
        )

    def define_displacement_boundary_condition_functions(self):
        # Get the dimensions of domain and facets
        dim = self.domain.mesh.geometry.dim
        fdim = self.domain.mesh.geometry.dim - 1
        # Get boundary facets
        boundary_facets = self.domain.boundary_facets
        # Get boundary dofs (per comp)
        boundary_dofs = {
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
        self.bcs_u = []
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
            self.bcs_u.append(
                fem.dirichletbc(
                    self.load_funcs[facet_name], boundary_dofs[facet_name], self.V_u
                )
            )

    def update_displacement_boundary_conditions(self, t: float):
        # Get displacement increments
        u_incs = self.pars["loading"]["u_incs"]
        # Iterate through the load functions
        for facet_name, load_func in self.load_funcs.items():
            # Increment the load function
            with load_func.vector.localForm() as bc_local:
                bc_local.set(default_scalar_type(t * u_incs[facet_name]))

    def define_problems(self):
        raise NotImplementedError(
            "Solver: The method 'define_problems' must be implemented in the child class."
        )

    def solve(self):
        print("\n████ RESOLUTION")
        # Start the loading iterations
        t_max = self.pars["loading"]["t_max"]
        for t in range(t_max + 1):
            # Display information
            print(f"== Load step {t}/{t_max}")
            # Update displacement boundary conditions
            self.update_displacement_boundary_conditions(t)
            # Solve the problems for this iteration
            self.solve_iteration()
            # Apply post processing
            self.postprocessor.postprocess()
            # Export the results
            self.exporter.export(t)
        # End export
        self.exporter.end_export()

    def solve_iteration(self):
        raise NotImplementedError(
            "Solver: The method 'solve_iteration' must be implemented in the child class."
        )
