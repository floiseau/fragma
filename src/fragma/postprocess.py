"""
Module for post-processing utilities.

This module provides classes and functions for post-processing simulation results.
"""

import dolfinx
from dolfinx import default_scalar_type, geometry, fem
import ufl

import numpy as np


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
        # Initialize dictionary for scalar data
        self.scalar_data = {}
        # Check the field to export
        fields = postprocess_pars.get("fields", {})
        # Initialize strain export
        if "strain" in fields:
            self.__initialize_strain(domain.mesh, model, state)
        # Initialize stress export
        if "stress" in fields:
            self.__initialize_stress(domain.mesh, model, state)
        # Initialize crack-driving variable field export
        if "crack-driving_variable" in fields:
            self.__initialize_crack_driving_variable(domain.mesh, model, state)
        # Initialize probes dict
        self.__initialize_probes(domain.mesh, state, postprocess_pars)
        # Initialize the reaction forces
        self.__initialize_reaction_forces(domain, model, state, postprocess_pars)
        # Initialize the energies computations
        self.__initialize_energies(domain, model, state)

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
        shape = eps_ufl.ufl_shape
        V_eps = fem.functionspace(mesh, ("DG", 0, shape))
        # Convert the strain into an expression
        self.exprs["eps"] = fem.Expression(eps_ufl, V_eps.element.interpolation_points)
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
        sig_ufl = model.sig_eff(state)
        # Generate FEM space for stress
        shape = sig_ufl.ufl_shape
        V_sig = fem.functionspace(mesh, ("DG", 0, shape))
        # Convert the stress into an expression
        self.exprs["sig"] = fem.Expression(sig_ufl, V_sig.element.interpolation_points)
        # Set the stress function
        self.funcs["sig"] = fem.Function(V_sig, name="Stress")
        self.funcs["sig"].interpolate(self.exprs["sig"])

    def __initialize_crack_driving_variable(self, mesh, model, state):
        """
        Initialize the crack-driving variables calculation.

        Parameters
        ----------
        mesh : dolfinx.Mesh
            The mesh representing the domain.
        model : BaseModel
            The material model used in the simulation.
        state : dict
            Dictionary containing state variables.
        """
        # Compute intermediate variables
        eps = model.eps(state)
        sig = model.sig(state)
        alpha = state["alpha"]
        # Compute the stress from ufl
        cdv_w = alpha * (1 - alpha)
        cdv_e = ufl.inner(sig, eps)
        cvd_ufl = cdv_w * cdv_e
        # Generate FEM space for crack-driving variable
        # V_cvd = fem.functionspace(mesh, ("Lagrange", 1))
        V_cvd = fem.functionspace(mesh, ("DG", 0))
        # Convert the crack-driving variable into an expression
        interp_points = V_cvd.element.interpolation_points
        self.exprs["cdv_e"] = fem.Expression(cdv_e, interp_points)
        self.exprs["cdv_w"] = fem.Expression(cdv_w, interp_points)
        self.exprs["cdv"] = fem.Expression(cvd_ufl, interp_points)
        # Set the crack-driving variable function
        self.funcs["cdv_e"] = fem.Function(V_cvd, name="Crack-driving Variable Energy")
        self.funcs["cdv_e"].interpolate(self.exprs["cdv_e"])
        self.funcs["cdv_w"] = fem.Function(V_cvd, name="Crack-driving Variable Weight")
        self.funcs["cdv_w"].interpolate(self.exprs["cdv_w"])
        self.funcs["cdv"] = fem.Function(V_cvd, name="Crack-driving Variable")
        self.funcs["cdv"].interpolate(self.exprs["cdv"])

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
                state["u"], np.array(displacement_probes_pos), mesh
            )

    def __initialize_reaction_forces(self, domain, model, state, postprocess_pars):
        """
        Initialize computation of reaction forces.

        Parameters
        ----------
        domain : Domain
            The domain containing the mesh and boundaries.
        model : Model
            The model containing the mathematical material model.
        state : dict
            Dictionary containing state variables.
        postprocess_pars : dict
            Dictionary containing parameters for post-processing.
        """
        # Get the dimension of the mesh
        dim = domain.mesh.geometry.dim
        # Get the surfaces on which to compute the reaction forces
        surfaces = postprocess_pars.get("reaction_forces", {})
        # Get the boundary facets from the domain
        boundary_facets = domain.boundary_facets
        # Compute the stress from ufl
        sig_ufl = model.sig_eff(state)
        # Initialize the dictionary of reaction forces expressions
        self.reaction_forces_forms = {}
        # Get the normals
        n = ufl.FacetNormal(domain.mesh)
        # Iterate through the surfaces
        for facet_name in surfaces:
            # Get the facets tags
            facet = boundary_facets[facet_name]
            facet_tags = dolfinx.mesh.meshtags(
                domain.mesh,
                domain.mesh.geometry.dim - 1,
                facet,
                np.full_like(facet, 1, dtype=np.int32),
            )
            # Get the associated integrand
            ds = ufl.Measure(
                "ds",
                domain=domain.mesh,
                subdomain_data=facet_tags,
                subdomain_id=1,
            )
            # Add the cohtribution to the external work
            for comp in range(dim):
                # Elementary vector
                elem_vec_np = np.zeros((dim,))
                elem_vec_np[comp] = 1
                elem_vec = fem.Constant(domain.mesh, elem_vec_np)
                # Set the expression of the reaction force along direction "comp"
                expr = ufl.dot(ufl.dot(sig_ufl, n), elem_vec) * ds
                # Get the associated form
                form = fem.form(expr)
                # Store the expression
                name = f"F_{comp + 1} ({facet_name})"
                self.reaction_forces_forms[name] = form
                self.scalar_data[name] = fem.assemble_scalar(form)

    def __initialize_energies(self, domain, model, state):
        """
        Initialize the computation of the energies.

        Parameters
        ----------
        model : fragma.domain.Domain
            The domain.
        model: BaseModel
            The material model.
        state : dict
            Dictionary containing state variables.
        """
        # Initialize the energy dictionary
        self.energies_forms = {}
        # Get the stored energies from the model
        if hasattr(model, "elastic_energy"):
            expr = model.elastic_energy(state, domain)
            self.energies_forms["elastic_energy"] = fem.form(expr)
        if hasattr(model, "fracture_dissipation"):
            expr = model.fracture_dissipation(state, domain)
            self.energies_forms["fracture_dissipation"] = fem.form(expr)
        # Undamaged elastic energy
        expr = fem.form(1 / 2 * ufl.inner(model.sig(state), model.eps(state)) * ufl.dx)
        self.energies_forms["undamaged_elastic_energy"] = fem.form(expr)
        # Computate of the external work
        u = state["u"]
        sig_ufl = model.sig_eff(state)
        n = ufl.FacetNormal(domain.mesh)
        ds = ufl.Measure("ds", domain=domain.mesh)
        expr = ufl.dot(ufl.dot(sig_ufl, n), u) * ds
        self.energies_forms["external_work"] = fem.form(expr)
        # Initialize the values
        for name, form in self.energies_forms.items():
            self.scalar_data[name] = fem.assemble_scalar(form)

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
        # Update the reaction forces
        for name, form in self.reaction_forces_forms.items():
            self.scalar_data[name] = fem.assemble_scalar(form)
        # Update the energies
        for name, expr in self.energies_forms.items():
            self.scalar_data[name] = fem.assemble_scalar(dolfinx.fem.form(expr))


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
