from dolfinx import fem, geometry
import ufl


class PostProcessor:
    def __init__(self, domain, model, state, postprocess_pars):
        """Initialize the post-processing."""
        # Initialize the post expressions and functions
        self.exprs = {}
        self.funcs = {}
        # Initialize strain export
        self.__initialize_strain(domain.mesh, model, state)
        # Initialize stress export
        self.__initialize_stress(domain.mesh, model, state)
        # Initialize the probes
        self.__initialize_stress(domain.mesh, model, state)
        # Initialize probes dict
        self.__initialize_probes(domain.mesh, state, postprocess_pars)

    def __initialize_strain(self, mesh, model, state):
        # Compute the strain from ufl
        eps_ufl = model.eps(state)
        # Generate FEM space for strain
        eps_elem = ufl.TensorElement("DG", mesh.ufl_cell(), 0, shape=eps_ufl.ufl_shape)
        V_eps = fem.FunctionSpace(mesh, eps_elem)
        # Convert the strain into an expression
        self.exprs["eps"] = fem.Expression(
            eps_ufl, V_eps.element.interpolation_points()
        )
        # Set the strain function
        self.funcs["eps"] = fem.Function(V_eps, name="Strain")
        self.funcs["eps"].interpolate(self.exprs["eps"])

    def __initialize_stress(self, mesh, model, state):
        # Compute the stress from ufl
        sig_ufl = model.sig(state)
        # Generate FEM space for stress
        sig_elem = ufl.TensorElement("DG", mesh.ufl_cell(), 0, shape=sig_ufl.ufl_shape)
        V_sig = fem.FunctionSpace(mesh, sig_elem)
        # Convert the stress into an expression
        self.exprs["sig"] = fem.Expression(
            sig_ufl, V_sig.element.interpolation_points()
        )
        # Set the stress function
        self.funcs["sig"] = fem.Function(V_sig, name="Stress")
        self.funcs["sig"].interpolate(self.exprs["sig"])

    def __initialize_probes(self, mesh, state, postprocess_pars):
        # Initialize the dict of probes
        self.probes = {}
        # Check if there are any probes
        probes_pars = postprocess_pars.get("probes", {})

        # Check if there are any displacement probes
        displacement_probes_pos = probes_pars.get("displacement", None)
        # Create the displacement probes
        if displacement_probes_pos is not None:
            print("Generate the displacement probes")
            self.probes["displacement"] = Probes(
                state["u"], displacement_probes_pos, mesh
            )

    def postprocess(self):
        """Update the post-processed quantities.

        This method computes the strain and stress fields in the mesh.
        """
        # Update the field functions
        for func, expr in zip(self.funcs.values(), self.exprs.values()):
            func.interpolate(expr)
        # Update the displacement probes values
        for probe in self.probes.values():
            probe.update()


class Probes:
    """Probes to evaluate func at the points xs."""

    def __init__(self, func, xs, mesh):
        """Initialize the displacement probes.

        This method is based on: https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html?#making-curve-plots-throughout-the-domain.
        Note that this source also contains the modifications for the parallel version.

        Input:
            func: Function to probe
            xs: Positions of the probe
        """
        # Store the function
        self.func = func
        # Get the position of the probes
        self.xs = xs
        # Generate the bounding box tree
        tree = geometry.bb_tree(mesh, mesh.topology.dim)
        # Find cells whose bounding-box collide with the the points
        cell_candidates = geometry.compute_collisions_points(tree, xs)
        # For each points, choose one of the cells that contains the point
        colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, xs)
        self.cells = [colliding_cells.links(i)[0] for i, x in enumerate(xs)]
        # Initialize the values
        self.vals = []
        # Initialize the probes values
        self.update()

    def update(self):
        self.vals = self.func.eval(self.xs, self.cells)


# NOTE: the following class is working but likely to be less efficient than Probes.
# class Probe:
#     """Probe to evaluate func at the point x."""
#
#     def __init__(self, func, x, mesh):
#         """Initialize a displacement probe.
#
#         This method is based on: https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html?#making-curve-plots-throughout-the-domain.
#         Note that this source also contains the modifications for the parallel version.
#
#         Input:
#             func: Function to probe
#             x: Position of the probe
#         """
#         # Store the function
#         self.func = func
#         # Get the position of the probes
#         self.x = x
#         # Generate the bounding box tree
#         bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
#         # Find cells whose bounding-box collide with the the points
#         cell_candidates = geometry.compute_collisions_points(bb_tree, [x])
#         # Choose one of the cells that contains the point
#         self.cell = geometry.compute_colliding_cells(mesh, cell_candidates, x)[0]
#         # Initialize the value
#         self.val = 0
#
#     def update(self):
#         self.val = self.func.eval([self.x], [self.cell])
#         print(self.val)
