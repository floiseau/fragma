from pathlib import Path
import json

from mpi4py import MPI

from dolfinx import io, fem, default_scalar_type


class Solver:
    def __init__(self, pars):
        ### Parameters
        print("\n████ PARAMETERS")
        # Store paramters
        self.pars = pars
        # Display a summary
        print(json.dumps(self.pars, indent=4))
        # Define the domain
        self.domain, cell_tags, self.facet_tags = self.define_domain()
        # Define the state variables
        self.define_state_variables()
        # Define the energy
        self.define_total_energy()
        # Define problems
        self.define_problems()
        # Start export
        self.init_export()

    def define_domain(self):
        print("\n████ DEFINITION OF THE DOMAIN")
        # Get the dimension
        dim = self.pars["model"]["dim"]
        # Read the mesh from GMSH
        msh_file = self.pars["mesh"]["msh_file"]
        print("Mesh reading output:")
        return io.gmshio.read_from_msh(msh_file, MPI.COMM_WORLD, gdim=dim)

    def define_state_variables(self):
        raise NotImplementedError(
            "Solver: The method 'define_state_variables' must be implemented in the child class."
        )

    def sig(self):
        raise NotImplementedError(
            "Solver: The method 'sig' must be implemented in the child class."
        )

    def eps(self):
        raise NotImplementedError(
            "Solver: The method 'eps' must be implemented in the child class."
        )

    def define_total_energy(self):
        raise NotImplementedError(
            "Solver: The method 'define_total_energy' must be implemented in the child class."
        )

    def define_displacement_boundary_condition_functions(self):
        ### Locate Boundary
        print("\n████ LOCATE BOUNDARIES")
        # Get the physical groups (mapping between pg and their indices)
        facets_tags_values = self.pars["mesh"]["physical_groups"]
        # Get the facets indices
        boundary_facets = {}
        for facet_name, facet_value in facets_tags_values.items():
            boundary_facets[facet_name] = self.facet_tags.indices[
                self.facet_tags.values == facet_value
            ]
        # Get the dimensions of domain and facets
        dim = self.domain.geometry.dim
        fdim = self.domain.geometry.dim - 1
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
                    self.load_funcs[facet_name], boundaries[facet_name], self.V_u
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
            # Export the results
            self.export_state(t)
        # End export
        self.end_export()

    def solve_iteration(self):
        raise NotImplementedError(
            "Solver: The method 'solve_iteration' must be implemented in the child class."
        )

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
        for state_field in self.state.values():
            self.xdmf_file.write_function(state_field, t)

    def end_export(self):
        # Close the file
        self.xdmf_file.close()
