"""
Module for post-processing utilities.

This module provides classes and functions for post-processing simulation results.
"""

from dolfinx import fem, geometry
import ufl


class PostProcessor:
    """
    Class for post-processing simulation results.

    This class provides functionalities to compute strain, stress, and other quantities from simulation results.

    Parameters
    ----------
    domain : Domain
        The domain object representing the computational domain.
    model : BaseModel
        The material model used in the simulation.
    state : dict
        Dictionary containing state variables.
    postprocess_pars : dict
        Dictionary containing parameters for post-processing.
    """

    def __init__(self, domain, model, state, postprocess_pars):
        """
        Initialize the PostProcessor.

        Parameters
        ----------
        domain : Domain
            The domain object representing the computational domain.
        model : BaseModel
            The material model used in the simulation.
        state : dict
            Dictionary containing state variables.
        postprocess_pars : dict
            Dictionary containing parameters for post-processing.
        """
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
        """
        Initialize strain calculation.

        Parameters
        ----------
        mesh : dolfinx.Mesh
            The mesh representing the domain.
        model : BaseModel
            The material model used in the simulation.
        state : dict
            Dictionary containing state variables.
        """
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
        """
        Initialize stress calculation.

        Parameters
        ----------
        mesh : dolfinx.Mesh
            The mesh representing the domain.
        model : BaseModel
            The material model used in the simulation.
        state : dict
            Dictionary containing state variables.
        """
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
        """
        Initialize probes.

        Parameters
        ----------
        mesh : dolfinx.Mesh
            The mesh representing the domain.
        state : dict
            Dictionary containing state variables.
        postprocess_pars : dict
            Dictionary containing parameters for post-processing.
        """
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
        """
        Perform post-processing.

        This method updates the post-processed quantities such as strain, stress, and probe values.
        """
        # Update the field functions
        for func, expr in zip(self.funcs.values(), self.exprs.values()):
            func.interpolate(expr)
        # Update the displacement probes values
        for probe in self.probes.values():
            probe.update()


class Probes:
    """
    Class to evaluate a function at specified points.

    This class represents probes used to evaluate a function at specific points in the domain.

    Parameters
    ----------
    func : dolfinx.Function
        The function to probe.
    xs : numpy.ndarray
        Positions of the probes.
    mesh : dolfinx.Mesh
        The mesh representing the domain.
    """

    def __init__(self, func, xs, mesh):
        """
        Initialize the Probes.

        This method is based on: https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html?#making-curve-plots-throughout-the-domain.
        Note that this source also contains the modifications for the parallel version.

        Parameters
        ----------
        func : dolfinx.Function
            The function to probe.
        xs : numpy.ndarray
            Positions of the probes.
        mesh : dolfinx.Mesh
            The mesh representing the domain.
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
        """Update the values of the probes."""
        self.vals = self.func.eval(self.xs, self.cells)
