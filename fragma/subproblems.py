"""
Module for defining sub-problems related to displacement and crack phase evolution.

This module provides classes for defining sub-problems that solve for displacement and crack phase evolution.
"""
from math import isnan

import numpy as np
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

from utils.build_nullspace import build_elasticity_nullspace
from utils.snes_problem import SNESProblem


def create_displacement_subproblem(pars, domain, state, model):
    """
    Create a displacement subproblem based on the provided parameters.

    Parameters
    ----------
    pars : dict
        Parameters for the subproblem.
    domain : Domain
        Domain object representing the computational domain.
    state : dict
        State variables for the problem.
    model : Model
        Model object defining the physics of the problem.

    Returns
    -------
    DisplacementSubProblem or DisplacementPartitionedSubProblem
        Depending on the loading constraint specified in the parameters, either a
        DisplacementSubProblem or a DisplacementPartitionedSubProblem instance is returned.

    Notes
    -----
    This function creates a displacement subproblem based on the parameters provided. If a loading
    constraint is specified in the parameters, a DisplacementPartitionedSubProblem instance is
    created to handle partitioned displacement problems. Otherwise, a DisplacementSubProblem
    instance is created for conventional displacement problems.

    Examples
    --------
    >>> subproblem = create_displacement_subproblem(pars, domain, state, model)
    """
    # Get the loading constraint
    constraint = pars.get("loading", {}).get("constraint", None)
    if constraint is None:
        print("Using the monolithic displacement sub-problem.")
        return DisplacementSubProblem(pars, domain, state, model)
    else:
        print("Using the partitioned displacement sub-problem.")
        return DisplacementPartitionedSubProblem(pars, domain, state, model)


class DisplacementSubProblem:
    """
    Class for solving the displacement sub-problem.

    This class defines a sub-problem that solves for displacement evolution.

    Parameters
    ----------
    pars : dict
        Dictionary containing parameters for the problem.
    domain : Domain
        The domain object representing the computational domain.
    state : dict
        Dictionary containing state variables.
    model : BaseModel
        The material model used in the simulation.
    """

    def __init__(self, pars, domain, state, model):
        """
        Initialize the DisplacementSubProblem.

        Parameters
        ----------
        pars : dict
            Dictionary containing parameters for the problem.
        domain : Domain
            The domain object representing the computational domain.
        state : dict
            Dictionary containing state variables.
        model : BaseModel
            The material model used in the simulation.
        """
        # Initialize the load factor
        self.l = 0.0
        # Store the displacement loading
        self.u_imp_max = pars["loading"].get("u_imp_max", {})
        # Store the force loading
        self.f_imp_max = pars["loading"].get("f_imp_max", {})
        # Store the contact force loading
        self.fc_max = pars["loading"].get("fc_max", {})
        # Check if t_max is defined
        if pars["end"].get("t_max", None):
            self.t_max = pars["end"]["t_max"]
        # Initialize the boundary conditions
        bcs_u = self.initialize_boundary_conditions(pars, domain, state)
        # Define the linear problem
        self.define_problem(domain, state, model, bcs_u)

    def initialize_boundary_conditions(self, pars, domain, state):
        """
        Initialize boundary conditions.

        Parameters
        ----------
        pars : dict
            Dictionary containing parameters for the problem.
        domain : Domain
            The domain object representing the computational domain.
        state : dict
            Dictionary containing state variables.

        Returns
        -------
        list
            List of boundary conditions for the displacement sub-problem.
        """
        bcs_u = []
        # Define a lock point
        bcs_u += self.define_lock_point(pars, domain, state)
        # Define the boundary conditions functions
        bcs_u += self.define_boundary_condition_functions(domain, state)
        # Return the boundary conditions
        return bcs_u

    def define_lock_point(self, pars, domain, state):
        """
        Define the lock point boundary condition.

        Parameters
        ----------
        pars : dict
            Dictionary containing parameters for the problem.
        domain : Domain
            The domain object representing the computational domain.
        state : dict
            Dictionary containing state variables.

        Returns
        -------
        list
            List of boundary conditions for the lock point.
        """
        # Get the position of the point
        x0 = pars.get("loading", {}).get("lock_point", None)
        # Return if there is not lock point
        if x0 is None:
            return []
        # Get the state variable
        u = state["u"]
        # Get the function space of the state variable
        V_u = u.function_space

        # Generate the location function
        def lock_point(x):
            return (
                np.isclose(x[0], x0[0])
                & np.isclose(x[1], x0[1])
                & np.isclose(x[2], x0[2])
            )

        # Generate the zero displacement vector
        u_zero = np.array((0,) * domain.mesh.geometry.dim, dtype=default_scalar_type)
        # Locate the dof
        dofs = fem.locate_dofs_geometrical(V_u, lock_point)
        # Generate the Dirichlet boundary condition
        return [fem.dirichletbc(u_zero, dofs, V_u)]

    def define_boundary_condition_functions(self, domain, state):
        """
        Define boundary condition functions for the displacement sub-problem.

        This method initializes the boundary conditions for the displacement sub-problem.

        Parameters
        ----------
        domain : Domain
            The domain object representing the computational domain.
        state : dict
            Dictionary containing state variables.

        Returns
        -------
        list
            List of boundary conditions for the displacement sub-problem.
        """
        print("\n████ DEFINITION OF THE DISPLACEMENT BOUNDARY CONDITIONS")
        # Get the state variable
        u = state["u"]
        # Get the dimensions of domain and facets
        dim = domain.mesh.geometry.dim
        fdim = domain.mesh.geometry.dim - 1
        # Get the displacement function space
        V_u = u.function_space
        # Get boundary facets
        boundary_facets = domain.boundary_facets
        # Get boundary dofs (per comp)
        boundary_dofs = {
            f"{facet_name}_{comp}": fem.locate_dofs_topological(
                (V_u.sub(comp), V_u.sub(comp).collapse()[0]),
                fdim,
                boundary_facet,
            )
            for comp in range(dim)
            for facet_name, boundary_facet in boundary_facets.items()
        }

        print("\n████ INITIALIZE DISPLACEMENT BOUNDARY CONDITIONS")
        # Create variables to store bcs and loading functions
        bcs_u = []
        self.bcu_funcs = {}
        # Iterage through the displacement loadings
        for facet_name, u_imp in self.u_imp_max.items():
            # Create a subdict for each components
            self.bcu_funcs[facet_name] = {}
            # Iterate through the axis
            for comp in range(dim):
                # Check if the DOF is imposed
                if isnan(self.u_imp_max[facet_name][comp]):
                    continue
                # Define an FEM function (to control the BC)
                self.bcu_funcs[facet_name][comp] = fem.Function(
                    V_u.sub(comp).collapse()[0]
                )
                # Update the load
                with self.bcu_funcs[facet_name][comp].vector.localForm() as bc_local:
                    bc_local.set(u_imp[comp])
                # Add the boundary conditions to the list
                bcs_u.append(
                    fem.dirichletbc(
                        self.bcu_funcs[facet_name][comp],
                        boundary_dofs[f"{facet_name}_{comp}"],
                        V_u,
                    )
                )
        return bcs_u

    def update_boundary_conditions(self, t: float):
        """
        Update boundary conditions for the displacement sub-problem.

        This method updates the displacement boundary conditions based on the current time.

        Parameters
        ----------
        t : float
            Current time.
        """
        # Iterate through the displacement load functions
        for facet_name, load_dict in self.bcu_funcs.items():
            # Iterate through the axis
            for comp, load_func in load_dict.items():
                # Check if the DOF is imposed
                if isnan(self.u_imp_max[facet_name][comp]):
                    continue
                # Update the load function
                with load_func.vector.localForm() as bc_local:
                    bc_local.set(
                        default_scalar_type(t * self.u_imp_max[facet_name][comp])
                    )
        # Iterate through the force load functions
        for facet_name, f_imp in self.f_imp_max.items():
            self.bcf_funcs[facet_name].value = t * np.array(f_imp)
        # Iterate through the contact force load functions
        for facet_name, fc in self.fc_max.items():
            F = fc["F"]
            self.bcf_funcs[facet_name].value = t * np.array(F)

    def compute_external_work(self, domain, state):
        """
        Compute the external work on the system.

        This method calculates the external work done on the system due to applied forces.
        It iterates through the boundary facets and computes the work done by each force.
        The total external work is obtained by summing up the work contributions from all the boundary facets.

        Parameters
        ----------
        domain : Domain
            The domain object representing the computational domain.
        state : dict
            Dictionary containing state variables.

        Returns
        -------
        ufl.Form
            The external work done on the system.

        Notes
        -----
        This method computes the external work by integrating the dot product of the applied forces
        and the displacement over the boundary facets of the domain.

        Examples
        --------
        >>> external_work = compute_external_work(domain, state)
        """
        # Get the state variable
        u = state["u"]
        # Get boundary facets
        boundary_facets = domain.boundary_facets
        # If there are not external forces
        if not self.f_imp_max and not self.fc_max:
            # Get the integrands
            ds = ufl.Measure("ds", domain=domain.mesh)
            # Initialize the external work
            T = fem.Constant(domain.mesh, [0.0] * domain.mesh.geometry.dim)
            # Return a null external work
            return ufl.dot(T, u) * ds
        # Otherwise initialize the external work
        external_work = 0.0
        # Initialize functions
        self.bcf_funcs = {}
        # Iterate through the forces
        for facet_name, f_imp in self.f_imp_max.items():
            # Get the facets tags
            facet = boundary_facets[facet_name]
            facet_tags = dolfinx.mesh.meshtags(
                domain.mesh,
                domain.mesh.geometry.dim - 1,
                facet,
                np.full_like(facet, 1, dtype=np.int32),
            )
            # Create the load function
            f = fem.Constant(domain.mesh, f_imp)
            self.bcf_funcs[facet_name] = f
            # Get the associated integrand
            ds = ufl.Measure(
                "ds",
                domain=domain.mesh,
                subdomain_data=facet_tags,
                subdomain_id=1,
            )
            # Add the cohtribution to the external work
            external_work += ufl.dot(f, u) * ds

        # Iterate through the forces
        for facet_name, fc in self.fc_max.items():
            # Get the parameters
            F = fem.Constant(domain.mesh, fc["F"])
            D = fc["D"]
            L = fc["L"]
            # Get the facets tags
            facet = boundary_facets[facet_name]
            facet_tags = dolfinx.mesh.meshtags(
                domain.mesh,
                domain.mesh.geometry.dim - 1,
                facet,
                np.full_like(facet, 1, dtype=np.int32),
            )
            # Store the load function
            self.bcf_funcs[facet_name] = F
            # Get the associated integrand
            ds = ufl.Measure(
                "ds",
                domain=domain.mesh,
                subdomain_data=facet_tags,
                subdomain_id=1,
            )
            # Compute the normal
            n = ufl.FacetNormal(domain.mesh)
            # Compute the pressure
            P = 4 / (np.pi * L * D) * ufl.dot(F, n)
            # Add the cohtribution to the external work
            external_work += P * ufl.dot(n, u) * ds
        return external_work

    def define_problem(self, domain, state, model, bcs_u):
        """
        Define the displacement problem.

        This method sets up the displacement problem by defining the energy and its derivatives with respect to
        the displacement variable. It then creates the LinearProblem object and initializes the PETSc linear solver.

        Parameters
        ----------
        domain : Domain
            The domain object representing the computational domain.
        state : dict
            Dictionary containing state variables.
        model : BaseModel
            The material model used in the simulation.
        bcs_u : list
            List of boundary conditions for the displacement sub-problem.
        """
        print("\n████ DEFINITION OF THE DISPLACEMENT PROBLEM")
        # Get the state variables
        u = state["u"]
        # Get the function spaces
        V_u = u.function_space
        # Define the total energy
        energy = model.energy(state, domain)
        external_work = self.compute_external_work(domain, state)
        if external_work:
            energy -= external_work
        # Derivative of the energy with respect to displacement to obtain the linear problem to determine the stationary point
        E_u = ufl.derivative(energy, u, ufl.TestFunction(V_u))
        E_du = ufl.replace(E_u, {u: ufl.TrialFunction(V_u)})
        # Define the displacement problem
        problem_u = LinearProblem(
            a=ufl.lhs(E_du),
            L=ufl.rhs(E_du),
            bcs=bcs_u,
            u=u,
            petsc_options={
                "ksp_type": "cg",
                "ksp_rtol": 1e-12,
                "ksp_atol": 1e-12,
                "ksp_max_it": 1000,
                "pc_type": "gamg",
                "pc_gamg_agg_nsmooths": 1,
                "pc_gamg_esteig_ksp_type": "cg",
            },
            # petsc_options={
            #     "ksp_type": "preonly",
            #     "pc_type": "lu",
            #     "pc_factor_solver_type": "mumps",
            # },
        )
        # Define the null space (optimization with GAMG PC)
        ns = build_elasticity_nullspace(V_u)
        problem_u.A.setNearNullSpace(ns)
        problem_u.A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore
        # Set block size
        problem_u.A.setBlockSize(V_u.mesh.geometry.dim)
        # Display information about the displacement solver
        problem_u.solver.view()
        # Store the problem
        self.problem_u = problem_u

    def update(self, t: float):
        """
        Update the displacement sub-problem.

        This method is typically used to update boundary conditions, problem bounds,
        right-hand side terms, or any other parameters that may change over time.

        Parameters
        ----------
        t : float
            Time parameter.
        """
        # Update the load factor
        self.l = t / self.t_max
        # Update boundary conditions
        self.update_boundary_conditions(self.l)

    def solve(self):
        """Solve the displacement sub-problem."""
        self.problem_u.solve()


class DisplacementPartitionedSubProblem(DisplacementSubProblem):
    """
    Class representing a partitioned displacement subproblem.

    This class extends the DisplacementSubProblem and implements a partitioned displacement
    problem, where the problem is defined and solved in terms of displacement increments.
    The resolution is carried by solving two linear system. More informations are available
    in the paper of Rastiello et al. [1].

    References
    ----------
    .. [1] Rastiello, G., Oliveira, H. L., & Millard, A. (2022). Path-following methods for
           unstable structural responses induced by strain softening: A critical review.
           Comptes Rendus. Mécanique, 350(G2), 205–236. https://doi.org/10.5802/crmeca.112
    """

    def __init__(self, pars, domain, state, model):
        """
        Initialize the partitioned displacement subproblem.

        Parameters
        ----------
        pars : dict
            Parameters for the problem.
        domain : Domain
            Domain of the problem.
        state : dict
            State variables of the problem.
        model : BaseModel
            Model defining the problem's behavior.
        """
        super().__init__(pars, domain, state, model)
        # Store the max load step
        if pars["end"]["criterion"] == "t":
            self.t_max = pars["end"]["t_max"]
        # Store the constraint
        self.constraint = pars["loading"]["constraint"]
        # Store the model
        self.model = model
        # Initialize the load factor
        self.l = 0.0
        # Constraint-specific initialization
        if self.constraint == "max_strain_inc":
            # Get the first load factor increment
            self.l0 = pars["loading"]["l0"]
            # Set the maximal increment of strain
            self.dtau = pars["loading"]["dtau"]
            # Generate a function space for strain-like scalars
            eps_ufl = model.eps(state)
            eps_elem = ufl.TensorElement(
                "DG", domain.mesh.ufl_cell(), 0, shape=eps_ufl.ufl_shape
            )
            self.V_eps = fem.FunctionSpace(domain.mesh, eps_elem)
            # Generate a function space for strain-like scalars
            eps_scal_elem = ufl.FiniteElement("DG", domain.mesh.ufl_cell(), 0)
            self.V_eps_scal = fem.FunctionSpace(domain.mesh, eps_scal_elem)

    def define_problem(self, domain, state, model, bcs_u):
        """
        Define the partitioned displacement problem.

        Note: The problem is defined and solved in terms of displacement increments.

        Parameters
        ----------
        domain : Domain
            Domain of the problem.
        state : dict
            State variables of the problem.
        model : BaseModel
            Model defining the problem's behavior.
        bcs_u : list
            List of boundary conditions for displacement.
        """
        # Store the displacement state variable
        self.u = state["u"]
        # Create the displacement at previous load step
        self.u0 = self.u.copy()
        # Define displacement functions
        self.ui = self.u.copy()
        self.u1 = self.u.copy()
        self.u2 = self.u.copy()
        # Generate an modified state with ui instead of u
        modified_state = state.copy()
        modified_state["u"] = self.ui
        # Define the problem using the parent class

        super().define_problem(domain, modified_state, model, bcs_u)

    def update(self, t: float):
        """
        Update the partitioned displacement subproblem.

        Parameters
        ----------
        t : float
            Time parameter.
        """
        # Store the time
        self.t = t
        # Reset the iteration counter
        self.k = 1
        # Store the displacement at the beginning of load step
        self.u0.x.array[:] = self.u.x.array
        self.u0.x.scatter_forward()
        # Check the constraint
        if self.constraint == "max_strain_inc":
            # Compute the normalized strain from previous load steps
            eps0 = self.model.eps({"u": self.u0})
            eps0_norm = ufl.sqrt(ufl.inner(eps0, eps0))
            eps0_normed_expr = fem.Expression(
                eps0 / eps0_norm, self.V_eps.element.interpolation_points()
            )
            self.eps0_normed = fem.Function(self.V_eps, name="NormedStrain")
            self.eps0_normed.interpolate(eps0_normed_expr)

    def solve(self):
        """Solve the partitioned displacement subproblem."""
        # Set boundary conditions to 0
        self.update_boundary_conditions(0.0)
        # Get the displacement increment
        self.problem_u.solve()
        self.ui.vector.copy(self.u1.vector)
        # Set boundary conditions to 1
        self.update_boundary_conditions(1.0)
        # Get the displacement increment
        self.problem_u.solve()
        self.ui.vector.copy(self.u2.vector)
        # Computation of the incremement of load factor
        match self.constraint:
            case "time":
                # Set the increment of load factor equal to t/t_max
                self.l = self.t / self.t_max
            case "max_strain_inc":
                if self.t > 1:
                    # Compute the load factor increment for each element
                    deps1 = self.model.eps({"u": self.u1 - self.u0})
                    deps2 = self.model.eps({"u": self.u2})
                    a0_expr = fem.Expression(
                        ufl.inner(self.eps0_normed, deps1),
                        self.V_eps_scal.element.interpolation_points(),
                    )
                    a0 = fem.Function(self.V_eps_scal, name="a0")
                    a0.interpolate(a0_expr)
                    a1_expr = fem.Expression(
                        ufl.inner(self.eps0_normed, deps2),
                        self.V_eps_scal.element.interpolation_points(),
                    )
                    a1 = fem.Function(self.V_eps_scal, name="a1")
                    a1.interpolate(a1_expr)
                    lambdas = (self.dtau - a0.x.array) / a1.x.array
                    # Choose the load factor using nested interval
                    a1_inf_0 = a1.x.array <= 0
                    a1_sup_0 = a1.x.array > 0
                    l_max = np.min(lambdas[a1_sup_0]) if any(a1_sup_0) else float("inf")
                    l_min = (
                        np.max(lambdas[a1_inf_0]) if any(a1_inf_0) else -float("inf")
                    )
                    # Check if the interval is valid
                    if l_max < l_min:
                        raise RuntimeError(
                            f"The maximal increment of load factor is inferior the the minimal (min: {l_min:.3g}, max: {l_max:.3g}.)"
                        )
                    # Choose the load factor as the upper bound
                    self.l = l_max
                elif self.t == 1:
                    # Arbitary load factor increment at first load step
                    self.l = self.l0
                elif self.t == 0:
                    # Arbitary load factor increment at first load step
                    self.l = 0
        # Update the displacement
        self.u.x.array[:] = self.u1.x.array + self.l * self.u2.x.array
        self.u.x.scatter_forward()
        # Increment the iteration counter
        self.k += 1


class CrackPhaseSubProblem:
    """
    Class for solving the crack phase sub-problem.

    This class defines a sub-problem that solves for crack phase evolution.

    Parameters
    ----------
    pars : dict
        Dictionary containing parameters for the problem.
    domain : Domain
        The domain object representing the computational domain.
    state : dict
        Dictionary containing state variables.
    model : BaseModel
        The material model used in the simulation.
    """

    def __init__(self, pars, domain, state, model):
        """
        Initialize the CrackPhaseSubProblem.

        Parameters
        ----------
        pars : dict
            Dictionary containing parameters for the problem.
        domain : Domain
            The domain object representing the computational domain.
        state : dict
            Dictionary containing state variables.
        model : BaseModel
            The material model used in the simulation.
        """
        # Define the boundary conditions functions
        bcs_alpha = self.define_boundary_condition_functions(domain, state)
        # Define the crack phase problem
        self.define_problem(domain, state, model, bcs_alpha)

    def define_problem(self, domain, state, model, bcs_alpha):
        """
        Define the crack phase problem.

        This method sets up the crack phase problem by defining the energy and its derivatives with respect to
        the crack phase variable. It then creates the SNESProblem object and initializes the PETSc SNES solver.

        Parameters
        ----------
        domain : Domain
            The domain object representing the computational domain.
        state : dict
            Dictionary containing state variables.
        model : BaseModel
            The material model used in the simulation.
        bcs_alpha : list
            List of boundary conditions for the crack phase sub-problem.
        """
        print("\n████ DEFINITION OF THE CRACK PHASE PROBLEM")
        # Get the state variables
        alpha = state["alpha"]
        # Store the state variable
        self.alpha = alpha
        # Get the function spaces
        V_alpha = alpha.function_space
        # Define the energy
        energy = model.energy(state, domain)
        # Derivative of the energy with respect to crack phase
        E_alpha = ufl.derivative(energy, alpha, ufl.TestFunction(V_alpha))
        E_alpha_alpha = ufl.derivative(E_alpha, alpha, ufl.TrialFunction(V_alpha))

        # Define the crack phase problem
        snes_problem_alpha = SNESProblem(E_alpha, E_alpha_alpha, alpha, bcs_alpha)
        # Initialize the LHS
        b = fem.petsc.create_vector(snes_problem_alpha.L)
        # Initialize the jacobian
        J = fem.petsc.create_matrix(fem.form(snes_problem_alpha.a))

        # Create the nonlinear solver
        problem_alpha = PETSc.SNES().create()
        problem_alpha.setType("vinewtonrsls")
        problem_alpha.setFunction(snes_problem_alpha.F, b)
        problem_alpha.setJacobian(snes_problem_alpha.J, J)
        problem_alpha.setTolerances(atol=1e-12, rtol=1e-12, max_it=50)

        # Set the KSP
        problem_alpha.getKSP().setType("gmres")
        problem_alpha.getKSP().setTolerances(atol=1e-15, rtol=1e-15)
        problem_alpha.getKSP().getPC().setType("mg")
        problem_alpha.getKSP().getPC().setMGLevels(1)

        # Define lower and upper bounds functions for the crack phase field
        self.alpha_lb = fem.Function(V_alpha, name="Lower bound")
        self.alpha_ub = fem.Function(V_alpha, name="Upper bound")
        # Set the lower bound
        with self.alpha_lb.vector.localForm() as alpha_lb_local:
            alpha_lb_local.set(0.0)
        fem.set_bc(self.alpha_lb.vector, bcs_alpha)
        # Set the upper bound
        one_alpha = fem.Function(V_alpha)
        with self.alpha_ub.vector.localForm() as alpha_ub_local:
            alpha_ub_local.set(1.0)
        fem.set_bc(self.alpha_ub.vector, bcs_alpha)
        # Set the crack phrase boundary bound (Note: they are passed as reference and not as values)
        problem_alpha.setVariableBounds(self.alpha_lb.vector, self.alpha_ub.vector)

        # Initialise alpha
        self.alpha_lb.vector.copy(alpha.vector)

        # Display information about the displacement solver
        problem_alpha.view()
        # Store the problem on alpha in subproblems
        self.problem_alpha = problem_alpha

    def define_boundary_condition_functions(self, domain, state):
        """
        Define boundary condition functions for the crack phase sub-problem.

        This method initializes the boundary conditions for the crack phase sub-problem.

        Parameters
        ----------
        domain : Domain
            The domain object representing the computational domain.
        state : dict
            Dictionary containing state variables.

        Returns
        -------
        list
            List of boundary conditions for the crack phase sub-problem.
        """
        print("\n████ INITIALISATION OF THE CRACK FIELD")
        # Add the damage boundary conditions if there is an initial crack
        if "crack" in domain.boundary_facets:
            # Get the dimensions of domain and facets
            dim = domain.mesh.geometry.dim
            fdim = domain.mesh.geometry.dim - 1
            # Get the crack phase function space
            V_alpha = state["alpha"].function_space
            # Get boundary facets
            boundary_facets = domain.boundary_facets
            # Get boundary dofs (per comp)
            boundaries = {
                f"{facet_name}": fem.locate_dofs_topological(
                    V_alpha,
                    fdim,
                    boundary_facet,
                )
                for facet_name, boundary_facet in boundary_facets.items()
            }
            # Create the crack boundary condition
            bcs_alpha_crack = [fem.dirichletbc(1.0, boundaries["crack"], V_alpha)]
            # Create the uncrackable crack boundary condition
            bcs_alpha_noncrackable = [
                fem.dirichletbc(0.0, b_dof, V_alpha)
                for boundary, b_dof in boundaries.items()
                if boundary.startswith("non-crackable")
            ]
            #
            bcs_alpha = bcs_alpha_crack + bcs_alpha_noncrackable
        else:
            bcs_alpha = []
        return bcs_alpha

    def update_boundary_conditions(self, t: float):
        """
        Update boundary conditions for the crack phase sub-problem.

        This method updates the crack phase boundary conditions based on the current time.

        Parameters
        ----------
        t : float
            Current time.
        """
        ...

    def update(self, t: float):
        """
        Update the crack phase sub-problem.

        This method updates the crack phase sub-problem at the specified time.

        Parameters
        ----------
        t : float
            Current time.
        """
        # Update the crack phase lower bound
        self.alpha.vector.copy(self.alpha_lb.vector)
        # Update of boundary conditions ?
        self.update_boundary_conditions(t)

    def solve(self):
        """Solve the crack phase sub-problem."""
        self.problem_alpha.solve(None, self.alpha.vector)
        self.alpha.x.scatter_forward()
