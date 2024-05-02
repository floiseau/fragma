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
        # Initialize probes dict
        self.__initialize_probes(domain.mesh, state, postprocess_pars)
        # Initialize the reaction forces
        self.__initialize_reaction_forces(domain, model, state, postprocess_pars)
        # Initialize the energies computations
        self.__initialize_energies(domain, model, state)
        # Initialize the SIFs computation
        self.__initialize_SIFs(domain, model, state, postprocess_pars)
        # Initialize the energy release rate computation
        self.__initialize_energy_release_rate(domain, model, state, postprocess_pars)
        # Initialize the T-stress computation
        self.__initialize_T_stress(domain, model, state, postprocess_pars)

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
        sig_ufl = model.sig_eff(state)
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
                name = f"F_{comp+1} ({facet_name})"
                self.reaction_forces_forms[name] = form
                self.scalar_data[name] = fem.assemble_scalar(form)

    def __initialize_energies(self, domain, model, state):
        """
        Initialize the computation of the energies.

        Parameters
        ----------
        mesh : dolfinx.Mesh
            The mesh representing the domain.
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

    def __initialize_SIFs(self, domain, model, state, postprocess_pars):
        """
        Initialize the computation of the stress intensity factors.

        This method is based on the interaction integral.
        The implementation of this method is based in [1]__ and the implementation of Pietro Gazzi.

        Parameters
        ----------
        domain : Domain
            The domain object containing the mesh and information on the boundaries.
        model: BaseModel
            The material model.
        state : dict
            Dictionary containing state variables.
        postprocess_pars : dict
            Dictionary containing parameters for post-processing.

        .. [1] De Lorenzis, L., Maurini, C. (2021). Basic computational methods for fracture mechanics. NEWFRAC Core School 2021 Course, Notebook 2-LEFM, https://gitlab.com/newfrac/CORE-school/newfrac-core-numerics/-/blob/master/02-LEFM.ipynb.
        """
        # Check if the energy release rate must be computated
        if "SIFs" not in postprocess_pars:
            return
        # Read the parameters
        xc = np.array(postprocess_pars["SIFs"]["crack_tip"])
        R_int = postprocess_pars["SIFs"]["R_int"]
        R_ext = postprocess_pars["SIFs"]["R_ext"]
        phi0 = np.deg2rad(postprocess_pars["SIFs"]["crack_growth_angle"])
        # Get the theta field
        theta_field = self.compute_theta_field(domain, xc, R_int, R_ext)
        theta = ufl.as_vector([ufl.cos(phi0), ufl.sin(phi0)]) * theta_field
        # Compute auxialiary displacement fields
        u_I_aux = self.compute_auxiliary_displacement_field(
            domain, model, xc, phi0, K_I_aux=1, K_II_aux=0
        )
        u_II_aux = self.compute_auxiliary_displacement_field(
            domain, model, xc, phi0, K_I_aux=0, K_II_aux=1
        )
        # Get the displacement field
        u = state["u"]
        # Compute the I-integrals
        I_I = self.compute_I_integral(domain, model, u, u_I_aux, theta)
        I_II = self.compute_I_integral(domain, model, u, u_II_aux, theta)
        # Compute Ep
        match model.assumption:
            case "plane_stress":
                Ep = model.E
            case "plane_strain":
                Ep = model.E / (1 - model.nu**2)
        # Store the forms
        self.K_I_form = fem.form(Ep / 2 * I_I)
        self.K_II_form = fem.form(Ep / 2 * I_II)
        # Compute the SIFs
        self.scalar_data["K_I"] = fem.assemble_scalar(self.K_I_form)
        self.scalar_data["K_II"] = fem.assemble_scalar(self.K_II_form)

    def compute_auxiliary_displacement_field(
        self, domain, model, xc, phi0, K_I_aux, K_II_aux
    ):
        # Get the polar coordinates
        x = ufl.SpatialCoordinate(domain.mesh)
        x_tip = ufl.as_vector(xc[:2])
        r_vec = x - x_tip
        r = ufl.sqrt(ufl.dot(r_vec, r_vec))
        theta = ufl.atan2(r_vec[1], r_vec[0]) - phi0
        # Get the elastic parameters
        mu = model.mu
        # Get kappa
        nu = model.nu
        match model.assumption:
            case "plane_stress":
                ka = (3 - nu) / (1 + nu)
            case "plane_strain":
                ka = 3 - 4 * nu
        # Compute the function f
        f_I, f_II = [0, 0], [0, 0]
        f_I[0] = (ka - ufl.cos(theta)) / (2 * mu) * ufl.cos(theta / 2)
        f_I[1] = (ka - ufl.cos(theta)) / (2 * mu) * ufl.sin(theta / 2)
        f_II[0] = (2 + ka + ufl.cos(theta)) / (2 * mu) * ufl.sin(theta / 2)
        f_II[1] = (2 - ka - ufl.cos(theta)) / (2 * mu) * ufl.cos(theta / 2)
        # Compute the displacement field
        ui = [
            ufl.sqrt(r / (2 * np.pi)) * (K_I_aux * f_I[i] + K_II_aux * f_II[i])
            for i in range(2)
        ]
        return ufl.as_vector(ui)

    def compute_I_integral(self, domain, model, u, u_aux, theta):
        # Compute the gradients
        grad_u = ufl.grad(u)
        grad_u_aux = ufl.grad(u_aux)
        # Compute the strains
        eps = ufl.sym(grad_u)
        eps_aux = ufl.sym(grad_u_aux)
        # Compute the stresses
        sig = model.sig({"u": u})
        sig_aux = model.sig({"u": u_aux})
        # Compute theta gradient and div
        div_theta = ufl.div(theta)
        grad_theta = ufl.grad(theta)
        # Compute the terms of the interaction integral
        dx = ufl.dx(domain=domain.mesh)
        Iw12 = 1 / 2 * ufl.inner(sig, eps_aux) * div_theta * dx
        Iw21 = 1 / 2 * ufl.inner(sig_aux, eps) * div_theta * dx
        Ig12 = ufl.inner(sig, grad_u_aux * grad_theta) * dx
        Ig21 = ufl.inner(sig_aux, grad_u * grad_theta) * dx
        # Compute the interaction integral expression
        I_expr = Ig12 + Ig21 - Iw12 - Iw21
        return I_expr

    def __initialize_energy_release_rate(self, domain, model, state, postprocess_pars):
        """
        Initialize the computation of the energy release rate.

        It contains the initialization of the $G-\theta$ method.
        The implementation of this method is based in [1]__ and the implementation of Pietro Gazzi.

        Parameters
        ----------
        domain : Domain
            The domain object containing the mesh and information on the boundaries.
        model: BaseModel
            The material model.
        state : dict
            Dictionary containing state variables.
        postprocess_pars : dict
            Dictionary containing parameters for post-processing.

        .. [1] De Lorenzis, L., Maurini, C. (2021). Basic computational methods for fracture mechanics. NEWFRAC Core School 2021 Course, Notebook 2-LEFM, https://gitlab.com/newfrac/CORE-school/newfrac-core-numerics/-/blob/master/02-LEFM.ipynb.
        """
        # Check if the energy release rate must be computated
        if "energy_release_rate" not in postprocess_pars:
            return
        # Check the dimension of the domain
        if domain.mesh.geometry.dim == 3:
            error_message = "The energy release rate can not be computed in 3D yet."
            error_message += (
                "To implemented it, a line should be specified in the parameter file."
            )
            error_message += "Then, the different theta regions are defined using cylinders around this line."
            raise NotImplementedError(
                "The energy release rate can not be computed in 3D yet."
            )
        # Read the parameters
        crack_tip = np.array(postprocess_pars["energy_release_rate"]["crack_tip"])
        R_int = postprocess_pars["energy_release_rate"]["R_int"]
        R_ext = postprocess_pars["energy_release_rate"]["R_ext"]
        alpha = np.deg2rad(
            postprocess_pars["energy_release_rate"]["crack_growth_angle"]
        )
        # Get the theta field
        theta_field = self.compute_theta_field(domain, crack_tip, R_int, R_ext)
        # Compute the energy release rate form
        u = state["u"]
        eps = model.eps(state)
        sig = model.sig_eff(state)
        theta_vector = ufl.as_vector([ufl.cos(alpha), ufl.sin(alpha)]) * theta_field
        dx = ufl.dx(domain=domain.mesh)
        G_expr = (
            ufl.inner(sig, ufl.grad(u) * ufl.grad(theta_vector)) * dx
            - 1 / 2 * ufl.inner(sig, eps) * ufl.div(theta_vector) * dx
        )
        self.G_form = fem.form(G_expr)
        # Initialize the
        self.scalar_data["G"] = fem.assemble_scalar(self.G_form)

    def compute_theta_field(self, domain, crack_tip, R_int, R_ext):
        """Determine the theta field.

        The theta field is equal to:
        - 1 when the distance to crack tip is below R_int
        - 0 when the distance to crack tip is over R_ext
        In between the two radii, the theta field smoothly transitions from 1 to 0.

        Parameters
        ----------
        domain : Domain
            The domain object containing the mesh and information on the boundaries.
        crack_tip : array-like
            Position of the crack tip.
        R_int : float
            Interior radius.
        R_ext : float
            Exterior radius.

        Returns
        -------
        theta_field : dolfinx.fem.Function
            FEM function containing the theta field.
        """

        # Define the distance to the crack tip
        def distance_to_crack_tip(x):
            return np.sqrt((x[0] - crack_tip[0]) ** 2 + (x[1] - crack_tip[1]) ** 2)

        # Define the variational problem to define theta
        V_theta = fem.functionspace(domain.mesh, ("Lagrange", 1))
        theta, theta_ = ufl.TrialFunction(V_theta), ufl.TestFunction(V_theta)
        a = ufl.dot(ufl.grad(theta), ufl.grad(theta_)) * ufl.dx
        L = (
            fem.Constant(domain.mesh, default_scalar_type(0.0))
            * theta_
            * ufl.dx(domain=domain.mesh)
        )
        # Set the boundary conditions
        # Imposing 1 in the inner circle and zero in the outer circle
        dofs_inner = fem.locate_dofs_geometrical(
            V_theta, lambda x: distance_to_crack_tip(x) <= R_int
        )
        dofs_out = fem.locate_dofs_geometrical(
            V_theta, lambda x: distance_to_crack_tip(x) >= R_ext
        )
        bc_inner = fem.dirichletbc(default_scalar_type(1.0), dofs_inner, V_theta)
        bc_out = fem.dirichletbc(default_scalar_type(0.0), dofs_out, V_theta)
        bcs = [bc_out, bc_inner]
        # Solve the problem
        problem = fem.petsc.LinearProblem(a, L, bcs=bcs)
        return problem.solve()

    def __initialize_T_stress(self, domain, model, state, postprocess_pars):
        """
        Initialize the computation of the energy release rate.

        It contains the initialization of the $G-\theta$ method.
        The implementation of this method is based in [1]__ and the implementation of Pietro Gazzi.

        Parameters
        ----------
        domain : Domain
            The domain object containing the mesh and information on the boundaries.
        model: BaseModel
            The material model.
        state : dict
            Dictionary containing state variables.
        postprocess_pars : dict
            Dictionary containing parameters for post-processing.

        .. [1] De Lorenzis, L., Maurini, C. (2021). Basic computational methods for fracture mechanics. NEWFRAC Core School 2021 Course, Notebook 2-LEFM, https://gitlab.com/newfrac/CORE-school/newfrac-core-numerics/-/blob/master/02-LEFM.ipynb.
        """
        # Check if the energy release rate must be computated
        if "T_stress" not in postprocess_pars:
            return
        # Check the dimension of the domain
        if domain.mesh.geometry.dim == 3:
            error_message = "The T-stress can not be computed in 3D yet."
            raise NotImplementedError(error_message)
        # Read the parameters
        crack_tip = np.array(postprocess_pars["T_stress"]["crack_tip"])
        R_int = postprocess_pars["T_stress"]["R_int"]
        R_ext = postprocess_pars["T_stress"]["R_ext"]
        alpha = np.deg2rad(postprocess_pars["T_stress"]["crack_growth_angle"])
        # Get the theta field
        theta_field = self.compute_theta_field(domain, crack_tip, R_int, R_ext)
        theta_vector = ufl.as_vector([ufl.cos(alpha), ufl.sin(alpha)]) * theta_field
        # Get the displacement field
        u = state["u"]
        # Get the elastic parameters
        E = model.E
        mu = model.mu
        # Other elastic parameter (not the bulk modulus !)
        nu = model.nu
        ka = (
            3 - 4 * model.nu
            if model.assumption == "plane_strain"
            else (3 - nu) / (1 + nu)
        )
        # Other parameters
        F = 1
        d = 1
        # Compute the auxiliary displacement field
        x = ufl.SpatialCoordinate(domain.mesh)
        x_tip = ufl.as_vector(crack_tip)
        r_vec = x - x_tip
        r = ufl.sqrt(ufl.dot(r_vec, r_vec))
        phi = ufl.atan2(r_vec[1], r_vec[0]) - alpha
        u_1_aux = (
            -F / np.pi * (ka + 1) / (8 * mu) * ufl.ln(r / d)
            - F / np.pi * 1 / (4 * mu) * ufl.sin(phi) ** 2
        )
        u_2_aux = -F / np.pi * (ka - 1) / (8 * mu) * phi + F / np.pi * 1 / (
            4 * mu
        ) * ufl.sin(phi) * ufl.cos(phi)
        u_aux = ufl.as_vector([u_1_aux, u_2_aux])
        # Compute displacement gradients
        grad_u = ufl.grad(u)
        grad_u_aux = ufl.grad(u_aux)
        # Compute the strains
        eps = ufl.sym(grad_u)
        eps_aux = ufl.sym(grad_u_aux)
        # Compute the stresses
        sig = model.sig_eff({"u": u})
        sig_aux = model.sig_eff({"u": u_aux})
        # Compute theta gradient and div
        div_theta = ufl.div(theta_vector)
        grad_theta = ufl.grad(theta_vector)
        # Compute the terms of the interaction integral
        dx = ufl.dx(domain=domain.mesh)
        Iw12 = 1 / 2 * ufl.inner(sig, eps_aux) * div_theta * dx
        Iw21 = 1 / 2 * ufl.inner(sig_aux, eps) * div_theta * dx
        Ig12 = ufl.inner(sig, grad_u_aux * grad_theta) * dx
        Ig21 = ufl.inner(sig_aux, grad_u * grad_theta) * dx
        # Compute the interaction integral expression
        I_expr = Ig12 + Ig21 - Iw12 - Iw21
        # Compute the T-stress value
        Ep = E
        Ep /= (1 - model.nu**2) if model.assumption == "plane_strain" else 1
        T_expr = Ep / F * I_expr
        # Compute the interaction integral form
        self.T_form = fem.form(T_expr)
        # Initialize the
        self.scalar_data["T_stress"] = fem.assemble_scalar(self.T_form)

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
        # Compute the energy release rate
        if "G" in self.scalar_data:
            self.scalar_data["G"] = fem.assemble_scalar(self.G_form)
        # Compute the T-stress
        if "T_stress" in self.scalar_data:
            self.scalar_data["T_stress"] = fem.assemble_scalar(self.T_form)
        # Compute SIFs
        if "K_I" in self.scalar_data:
            self.scalar_data["K_I"] = fem.assemble_scalar(self.K_I_form)
        if "K_II" in self.scalar_data:
            self.scalar_data["K_II"] = fem.assemble_scalar(self.K_II_form)


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
