"""
Module for defining sub-problems related to displacement and crack phase evolution.

This module provides classes for defining sub-problems that solve for displacement and crack phase evolution.
"""

from math import isnan

import numpy as np
from scipy.optimize import root_scalar
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

from .utils.parameter_parser import parse_boundary_condition
from .utils.build_nullspace import build_elasticity_nullspace
from .utils.petsc_problems import SNESProblem, TAOProblem


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
        # Store the model
        self.model = model
        # Initialize the load factor
        self.l = pars["loading"].get("l0", 0.0)
        # Store varying displacement loading
        self.u_imp_max = pars["loading"].get("u_imp_max", {})
        # Store time-controlled displacement loading
        self.tc_u = pars["loading"].get("time-controlled_u", {})
        # Store constant displacement loading
        self.u_imp_const = pars["loading"].get("u_imp_const", {})
        # Store the force loading
        self.f_imp_max = pars["loading"].get("f_imp_max", {})
        # Check if t_max is defined
        if "dl" in pars["loading"]:
            self.dl = pars["loading"]["dl"]
        elif "t_max" in pars["end"]:
            self.dl = 1 / pars["end"]["t_max"]
        # Initialize the boundary conitions
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
        # Define boundary conditions for prescribed nodal displacements
        bcs_u += self.define_prescribed_nodal_displacements(pars, domain, state)
        # Define the boundary conditions functions
        bcs_u += self.define_boundary_condition_functions(domain, state)
        # Return the boundary conditions
        return bcs_u

    def define_prescribed_nodal_displacements(self, pars, domain, state):
        """
        Define the boundary conditions to prescribe nodal displacement.

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
            List of boundary conditions for the prescribed nodal displacement.
        """
        # Get the position of the point
        nod_disp_dict = pars.get("loading", {}).get("nodal_displacement", None)
        # Return if there is no imposed nodal displacement
        if nod_disp_dict is None:
            return []
        # Initialize the list of boundary conditions
        bcs = []
        # Get the state variable
        u = state["u"]
        # Get the function space of the state variable
        V_u = u.function_space
        # Iterate through the imposed nodal displacements
        for nd in nod_disp_dict.values():
            # Get the position
            x_imp = nd["x"]
            u_imp = nd["u"]

            # Generate the location function
            def lock_point(x):
                return (
                    np.isclose(x[0], x_imp[0])
                    & np.isclose(x[1], x_imp[1])
                    & np.isclose(x[2], x_imp[2])
                )

            # Iterate through the components
            for comp, val in enumerate(u_imp):
                # Check if the value is nan
                if isnan(val):
                    continue
                # Add the new bc
                dofs = fem.locate_dofs_geometrical((V_u.sub(comp), V_u), lock_point)[0]
                u_val = fem.Constant(domain.mesh, default_scalar_type(val))
                new_bc = fem.dirichletbc(u_val, dofs, V_u.sub(comp))
                bcs.append(new_bc)
        return bcs

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

        print("\n████ INITIALIZE VARYING DISPLACEMENT BOUNDARY CONDITIONS")
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
                func = self.bcu_funcs[facet_name][comp]
                with func.x.petsc_vec.localForm() as bc_local:
                    bc_local.set(u_imp[comp])
                # Add the boundary conditions to the list
                bcs_u.append(
                    fem.dirichletbc(
                        self.bcu_funcs[facet_name][comp],
                        boundary_dofs[f"{facet_name}_{comp}"],
                        V_u,
                    )
                )

        print("\n████ INITIALIZE CONSTANT DISPLACEMENT BOUNDARY CONDITIONS")
        # Iterage through the displacement loadings
        for facet_name, u_imp in self.u_imp_const.items():
            # Create a subdict for each components
            self.bcu_funcs[facet_name] = {}
            # Iterate through the axis
            for comp in range(dim):
                # Check if the DOF is imposed
                if isnan(self.u_imp_const[facet_name][comp]):
                    continue
                # Define an FEM function (to control the BC)
                func = fem.Function(V_u.sub(comp).collapse()[0])
                # Update the load
                with func.x.petsc_vec.localForm() as bc_local:
                    bc_local.set(u_imp[comp])
                # Add the boundary conditions to the list
                bcs_u.append(
                    fem.dirichletbc(
                        func,
                        boundary_dofs[f"{facet_name}_{comp}"],
                        V_u,
                    )
                )

        print("\n████ INITIALIZE TIME-CONTROLLED DISPLACEMENT BOUNDARY CONDITIONS")
        # Create variables to store bcs and loading functions
        self.tc_bcu_exprs_funcs = {}
        # Iterage through the displacement loadings
        for facet_name, u_imp in self.tc_u.items():
            # Create a subdict for each components
            self.tc_bcu_exprs_funcs[facet_name] = {}
            # Iterate through the axis
            for comp in range(dim):
                # Check if the DOF is imposed
                bc_par = self.tc_u[facet_name][comp]
                if isinstance(bc_par, float) and isnan(bc_par):
                    continue
                # Define an FEM function (to control the BC)
                func = fem.Function(V_u.sub(comp).collapse()[0])
                # Initialize the load
                bc_expr = parse_boundary_condition(u_imp[comp])
                # Create the fem function
                func.interpolate(lambda xx: bc_expr(xx, 0))
                # Add the boundary conditions to the list
                bcs_u.append(
                    fem.dirichletbc(
                        func,
                        boundary_dofs[f"{facet_name}_{comp}"],
                        V_u,
                    )
                )
                # Store the expression and the function
                self.tc_bcu_exprs_funcs[facet_name][comp] = (bc_expr, func)

        return bcs_u

    def update_boundary_conditions(self, l: float):
        """
        Update boundary conditions for the displacement sub-problem.

        This method updates the displacement boundary conditions based on the current time.

        Parameters
        ----------
        l : float
            Load factor.
        """
        # Iterate through the displacement load functions
        for facet_name, load_dict in self.bcu_funcs.items():
            # Iterate through the axis
            for comp, load_func in load_dict.items():
                # Check if the DOF is imposed
                if isnan(self.u_imp_max[facet_name][comp]):
                    continue
                # Update the load function
                with load_func.x.petsc_vec.localForm() as bc_local:
                    bc_local.set(
                        default_scalar_type(l * self.u_imp_max[facet_name][comp])
                    )
                load_func.x.scatter_forward()
        # Iterate through the time-controlled boundary conditions
        for facet_name, load_dict in self.tc_bcu_exprs_funcs.items():
            # Iterate through the components
            for comp, (expr, func) in load_dict.items():
                # Check if the DOF is imposed
                bc_par = self.tc_u[facet_name][comp]
                if isinstance(bc_par, float) and isnan(bc_par):
                    continue
                else:
                    func.interpolate(lambda xx: expr(xx, l))
        # Iterate through the force load functions
        for facet_name, f_imp in self.f_imp_max.items():
            self.bcf_funcs[facet_name].value = l * np.array(f_imp)
        # Update potential thermal load
        if self.model.thermal_load:
            # Update the temperature field
            self.model.dT.interpolate(lambda x: self.model.dT_lambda(x, l))

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
        if not self.f_imp_max:
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
        # Create the mesh tags
        facets = np.concatenate(
            [boundary_facets[facet_name] for facet_name in self.f_imp_max],
            dtype=np.int32,
        )
        facet_ids = np.concatenate(
            [
                [id + 1] * len(boundary_facets[facet_name])
                for id, facet_name in enumerate(self.f_imp_max)
            ],
            dtype=np.int32,
        )
        facet_tags = dolfinx.mesh.meshtags(
            domain.mesh,
            domain.mesh.geometry.dim - 1,
            facets,
            facet_ids,
        )
        # Get the surface integrand
        ds = ufl.Measure("ds", domain=domain.mesh, subdomain_data=facet_tags)
        # Apply the forces
        for id, (facet_name, f_imp) in enumerate(self.f_imp_max.items()):
            # Create the load function
            f = fem.Constant(domain.mesh, np.array(f_imp, dtype=default_scalar_type))
            self.bcf_funcs[facet_name] = f
            # Add the cohtribution to the external work
            external_work += ufl.dot(f, u) * ds(id + 1)

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
        # Define PETSc solver
        direct_solver = True
        if direct_solver:
            petsc_options = {
                "ksp_type": "preonly",
                "pc_type": "cholesky",
                "pc_factor_mat_solver_type": "cholmod",
                # "pc_type": "cholesky",
                # "pc_factor_mat_solver_type": "mumps",
            }
        else:
            petsc_options = {
                "ksp_type": "cg",
                "ksp_rtol": 1e-9,
                "ksp_atol": 1e-10,
                "ksp_max_it": 1000,
                "pc_type": "gamg",
                "pc_gamg_agg_nsmooths": 1,
                "pc_gamg_esteig_ksp_type": "cg",
                "pc_gamg_type": "agg",  # Aggressive coarsening
                "mg_levels_ksp_type": "chebyshev",
                "mg_levels_pc_type": "jacobi",
            }

        # Define the displacement problem
        problem_u = LinearProblem(
            a=ufl.lhs(E_du),
            L=ufl.rhs(E_du),
            bcs=bcs_u,
            u=u,
            petsc_options_prefix="elastic_linear_problem",
            petsc_options=petsc_options,
        )
        # Define the null space (optimization with GAMG PC)
        ns = build_elasticity_nullspace(V_u)
        problem_u.A.setNearNullSpace(ns)
        problem_u.A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore
        # Set block size
        problem_u.A.setBlockSize(V_u.mesh.geometry.dim)
        # Set nonzero initial guess
        if not direct_solver:
            problem_u.solver.setInitialGuessNonzero(True)
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
        self.l += self.dl if t > 0 else 0
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
        # Store the constraint
        self.constraint = pars["loading"]["constraint"]
        # Initialize the load factor
        self.l0 = pars["loading"].get("l0", 0.0)
        self.l = pars["loading"].get("l0", 0.0)
        # Initialize the iteration counter
        self.k = 1
        # Get the load factor increment in the initial phase
        self.dl = pars["loading"]["dl"]
        if self.constraint != "load_factor_inc":
            # Set the step size
            self.dtau = pars["loading"]["dtau"]
            # Set the minimal step size (during crack propagation)
            self.dtau_min = pars["loading"].get("dtau_min", 0)
        # Constraint-specific initialization
        match self.constraint:
            case "max_strain_inc" | "max_strain_inc_outside_crack":
                # Generate a function space for strain-like scalars
                eps_ufl = model.eps(state)
                shape = eps_ufl.ufl_shape
                V_eps = fem.functionspace(domain.mesh, ("DG", 0, shape))
                # Generate a function space for strain-like scalars
                V_eps_scal = fem.functionspace(domain.mesh, ("DG", 0))
                # Define the normed strain expression from previous load steps
                eps0 = self.model.eps({"u": self.u0})
                eps0_norm = ufl.sqrt(ufl.inner(eps0, eps0))
                self.eps0_normed_expr = fem.Expression(
                    eps0 / eps0_norm, V_eps.element.interpolation_points
                )
                self.eps0_normed = fem.Function(V_eps, name="NormedStrain")
                # Get the strain increment values
                deps1 = self.model.eps({"u": self.u1 - self.u0})
                deps2 = self.model.eps({"u": self.u2})
                # Define the coefficients expressions and functions
                a0_ufl = ufl.inner(self.eps0_normed, deps1)
                a1_ufl = ufl.inner(self.eps0_normed, deps2)
                if self.constraint == "max_strain_inc_outside_crack":
                    a0_ufl *= 1 - self.alpha0 + 1e-12
                    a1_ufl *= 1 - self.alpha0 + 1e-12
                self.a0_expr = fem.Expression(
                    a0_ufl, V_eps_scal.element.interpolation_points
                )
                self.a1_expr = fem.Expression(
                    a1_ufl, V_eps_scal.element.interpolation_points
                )
                self.a0 = fem.Function(V_eps_scal, name="a0")
                self.a1 = fem.Function(V_eps_scal, name="a1")

            case "nodal_disp_inc":
                # Get the parameters
                self.selection = pars["loading"]["selection"]
                # Compute the selection matrix
                c = np.hstack([p["coeff"] for p in self.selection])
                self.s = np.einsum("i,j->ij", c, c)
                # Get the cells for displacement evaluation
                mesh = domain.mesh
                xs = np.array([p["pos"] for p in self.selection])
                tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
                cell_candidates = dolfinx.geometry.compute_collisions_points(tree, xs)
                colliding_cells = dolfinx.geometry.compute_colliding_cells(
                    mesh, cell_candidates, xs
                )
                cells = [colliding_cells.links(i)[0] for i, x in enumerate(xs)]
                # Store the cells in selection
                for i, p in enumerate(self.selection):
                    p["cell"] = cells[i]

            case "local_arc_length":
                # Get the functions
                u = self.u2
                u0 = self.u0
                # Define a function space for displacements dot product
                V_u_scal = fem.functionspace(domain.mesh, ("Lagrange", 1))
                # Define the norm of the initial displacement field
                u0_norm = ufl.sqrt(ufl.dot(u0, u0))
                # Get the coefficients expressions
                self.a0_expr = fem.Expression(
                    -ufl.dot(u0, u0) / u0_norm,
                    V_u_scal.element.interpolation_points,
                )
                self.a1_expr = fem.Expression(
                    ufl.dot(u, u0) / u0_norm,
                    V_u_scal.element.interpolation_points,
                )
                # Define the functions
                self.a0 = fem.Function(V_u_scal, name="a0")
                self.a1 = fem.Function(V_u_scal, name="a1")

            case "crack_driving_variable":
                # Initialize the previous tau
                self.tau0 = 0
                # Get crack phase
                alpha = self.alpha
                # Compute the strain
                eps2 = self.model.eps({"u": self.u2})
                # Compute the stress
                sig2 = self.model.sig({"u": self.u2})
                # Define the expressions
                weight_expr = self.model.ap(alpha) * alpha
                energy_expr = 1 / 2 * ufl.inner(eps2, sig2)
                # Define the form
                dx = ufl.Measure("dx", domain=domain.mesh)
                self.tau2_form = fem.form(-weight_expr * energy_expr * dx)
                # Initialize tau2
                self.tau2 = fem.assemble_scalar(self.tau2_form)

            case "maxt_crack_driving_variable":
                # Generate a function space for history fields
                V_H = fem.functionspace(domain.mesh, ("DG", 0))
                # Define the functions
                self.H0 = fem.Function(V_H, name="H0")
                # self.l_min_func = fem.Function(V_H, name="H0")
                # Define an fem constant for lambda
                self.l_const = fem.Constant(domain.mesh, self.l)
                # Define the form to compute H_bar
                alpha0 = self.alpha0
                eps2 = self.model.eps({"u": self.u2})
                sig2 = self.model.sig({"u": self.u2})
                H_bar = (
                    -1
                    / 2
                    * self.l_const**2
                    * self.model.ap(alpha0)
                    * alpha0
                    * ufl.inner(eps2, sig2)
                )
                # Define the residual functional
                dx = ufl.Measure("dx", domain=domain.mesh)
                dH_expr = ufl.max_value(H_bar - self.H0, 0) * dx
                self.dH_form = fem.form(dH_expr)
                # # Define the expression to obtain the minimum load factor
                # l_min_ufl = self.H0 / (
                #     -1 / 2 * self.model.ap(alpha) * alpha * ufl.inner(eps2, sig2)
                # )
                # self.l_min_expr = fem.Expression(
                #     l_min_ufl, V_H.element.interpolation_points()
                # )
                # Define the expression to update H0
                self.H0_update_expr = fem.Expression(
                    ufl.max_value(H_bar, self.H0), V_H.element.interpolation_points
                )

        # Initialization of the step size adapation
        self.step_size_adapation = "k_opt" in pars["loading"]
        if self.step_size_adapation:
            # Store the optimal number of alternate minimization iterations
            self.k_opt = pars["loading"]["k_opt"]
            # Disable the step size adapation at the begining
            self.step_size_adapation_enabled = False

    def define_problem(self, domain, state, model, bcs_u):
        """
        Define the partitioned displacement problem.

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
        # Store the crack phase variable
        self.alpha = state.get("alpha", None)
        self.alpha0 = state.get("alpha0", None)
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
        # Store the number of iteration to converge in previous load step
        self.k_nm1 = self.k
        # Reset the iteration counter
        self.k = 1
        # Select the control equation for the current load step
        self.control_eq = self.select_control_equation()
        # Adapt the step size if the option is enabled
        if self.step_size_adapation:
            self.adapt_step_size()
        # Store the displacement at the beginning of load step
        self.u0.x.array[:] = self.u.x.array
        self.u0.x.scatter_forward()
        # Store the load factor at the beginning of the load step
        self.l0 = self.l
        # Check the constraint
        match self.constraint:
            case "max_strain_inc" | "max_strain_inc_outside_crack":
                # Update the previous normed strain field
                self.eps0_normed.interpolate(self.eps0_normed_expr)
            case "crack_driving_variable":
                # Update the previous tau value
                self.tau2 = fem.assemble_scalar(self.tau2_form)
                self.tau0 = self.l**2 * self.tau2

            case "maxt_crack_driving_variable":
                # Update the previous history field
                self.H0.interpolate(self.H0_update_expr)

    def select_control_equation(self):
        """Select the control equation.

        This method selects the load control equation based on the specified constraint
        and the current phase of the simulation.

        Returns
        -------
        str
            The selected control equation.
        """
        return self.constraint if self.t > 1 else "load_factor_inc"

    def adapt_step_size(self):
        """Adapt the step size during the simulation.

        This method adjusts the step size (`dtau`) based on the control equation and
        the current iteration number. If the step size adaptation flag is set to
        False and the iteration number exceeds the optimal number of AM iterations, the flag
        is set to True to indicate that step size adaptation should begin. If step size
        adaptation is active, the step size is adjusted by multiplying it by the ratio of
        the optimal iteration number to the iteration number of the previous load step.
        """
        # Check the step size adapation must be started
        if not self.step_size_adapation_enabled:
            self.step_size_adapation_enabled = self.t > 2
            self.step_size_adapation_enabled &= self.k_nm1 > self.k_opt / 2
        # Adapt the step size dtau
        if self.step_size_adapation_enabled:
            # Compute and apply the adaptation coefficient
            coeff = self.k_opt / self.k_nm1
            self.dtau *= coeff
            # If the load factor decrease (crack propagating), apply a min value
            if self.l0 > self.l:
                self.dtau = max(self.dtau_min, self.dtau)
            print(f"Step size adapation: dtau={self.dtau:.3g}.")

    def solve(self):
        """Solve the partitioned displacement subproblem."""
        # Set boundary conditions to 0
        self.update_boundary_conditions(0.0)
        # Solve the prescribed displacement problem
        self.ui.x.array[:] = self.u1.x.array[:]
        self.problem_u.solve()
        self.u1.x.array[:] = self.ui.x.array
        # Set boundary conditions to 1
        self.update_boundary_conditions(1.0)
        # Solve the controlled displacement problem
        self.ui.x.array[:] = self.u2.x.array
        self.problem_u.solve()
        self.u2.x.array[:] = self.ui.x.array
        # Computation of the incremement of load factor
        match self.control_eq:
            case "load_factor_inc":
                if self.k == 1 and self.t >= 1:
                    # Increment the load factor
                    self.l += self.dl

            case "max_strain_inc" | "max_strain_inc_outside_crack" | "local_arc_length":
                self.l = self.nested_interval(self.a0_expr, self.a1_expr)

            case "nodal_disp_inc":
                # Compute the displacement vector for u2
                uc2_list = [self.u2.eval(p["pos"], p["cell"]) for p in self.selection]
                uc2 = np.hstack(uc2_list)
                # Compute the displacement vector for u1
                uc1_list = [self.u1.eval(p["pos"], p["cell"]) for p in self.selection]
                uc1 = np.hstack(uc1_list)
                # Compute the displacement vector for u0
                uc0_list = [self.u0.eval(p["pos"], p["cell"]) for p in self.selection]
                uc0 = np.hstack(uc0_list)
                # Compute the control terms
                du22 = np.einsum("i,ij,j", uc2, self.s, uc2)
                du12 = np.einsum("i,ij,j", uc1 - uc0, self.s, uc2)
                du11 = np.einsum("i,ij,j", uc1 - uc0, self.s, uc1 - uc0)
                # Compute the polynomial coefficients
                a = du22
                b = 2 * du12
                c = du11 - self.dtau**2
                d = b**2 - 4 * a * c
                # Compute the load factor
                self.l = (-b + np.sqrt(d)) / (2 * a)

            case "crack_driving_variable":
                # Compute tau2
                self.tau2 = fem.assemble_scalar(self.tau2_form)
                # Compute the load factor
                self.l = np.sqrt((self.dtau + self.tau0) / self.tau2)

            case "maxt_crack_driving_variable":
                # Define a function to compute the history field increment
                def f(l):
                    self.l_const.value = l
                    fun = fem.assemble_scalar(self.dH_form) - self.dtau
                    return fun

                # Find max bracket with positive sign
                l_min = 0
                l_max = 2 * self.l
                while f(l_max) < 0:
                    l_min = l_max
                    l_max *= 2

                # Find the load factor
                res = root_scalar(f, bracket=(l_min, l_max), method="toms748")
                if not res.converged:
                    raise RuntimeError(
                        "Non-linear solver for path-following constraint (maxt_crack_driving_variable) failed to converged!"
                    )
                # Compute the load factor
                self.l = res.root
                # Update the constant value
                self.l_const.value = self.l

        # Update the displacement
        self.u.x.array[:] = self.u1.x.array + self.l * self.u2.x.array
        self.u.x.scatter_forward()
        # Increment the iteration counter
        self.k += 1

    def nested_interval(self, a0_expr, a1_expr):
        # Initialize l_min and l_max
        l_min = float("inf")
        l_max = -float("inf")
        # Define a counter
        dtau = self.dtau
        max_iter = 1
        iter = 0
        # Check if the interval is valid
        while l_max < l_min:
            # Update the coefficients
            self.a0.interpolate(a0_expr)
            self.a1.interpolate(a1_expr)
            lambdas = (dtau - self.a0.x.array) / self.a1.x.array
            # Choose the load factor using nested interval
            a1_inf_0 = self.a1.x.array <= 0
            a1_sup_0 = self.a1.x.array > 0
            l_max = np.min(lambdas[a1_sup_0]) if any(a1_sup_0) else float("inf")
            l_min = np.max(lambdas[a1_inf_0]) if any(a1_inf_0) else -float("inf")
            # Increment the iteration counter
            iter += 1
            # Divide dtau by 2 (used in the case l_max < l_min)
            dtau /= 2
            # Stop the loop if there are too much iterations
            if iter > max_iter:
                raise RuntimeError(
                    f"The maximum load factor ({l_max=:.3g}) is inferior the the minimal one ({l_min=:.3g}) after {max_iter} iterations."
                )

        # Choose the load factor as the upper bound
        return l_max


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
        # Get the solver type
        self.petsc_solver = pars["numerical"].get("nl_solver", "snes")
        # Get solver parameters
        solver_pars = pars["numerical"].get("problem_alpha", {})
        if self.petsc_solver == "snes":
            self.solver_tols = {
                "atol": solver_pars.get("atol", 1e-7),
                "rtol": solver_pars.get("rtol", 1e-7),
                "stol": solver_pars.get("stol", 1e-7),
                "max_it": solver_pars.get("max_it", 100),
            }

        elif self.petsc_solver == "tao":
            self.solver_tols = {
                "gatol": solver_pars.get("gatol", 1e-6),
                "grtol": solver_pars.get("grtol", 1e-6),
                "gttol": solver_pars.get("gttol", 1e-6),
            }
        # Define the boundary conditions functions
        bcs_alpha = self.define_boundary_condition_functions(domain, state)
        # Define the initial crack field
        self.define_initial_crack_field(pars, domain, state, model, bcs_alpha)
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

        # Initialize PETSc options
        opts = PETSc.Options()
        opts.setValue("snes_monitor", None)
        # opts.setValue("snes_vi_monitor", None)
        opts.setValue("snes_converged_reason", None)
        opts.setValue("tao_monitor", None)
        opts.setValue("tao_converged_reason", None)

        # Choose the solver for the bounded nonlinear problem
        if self.petsc_solver == "snes":
            # Define the crack phase problem
            snes_problem_alpha = SNESProblem(E_alpha, E_alpha_alpha, alpha, bcs_alpha)

            # Create the nonlinear solver
            problem_alpha = PETSc.SNES().create(MPI.COMM_WORLD)
            problem_alpha.setFunction(snes_problem_alpha.F, snes_problem_alpha.b)
            problem_alpha.setJacobian(snes_problem_alpha.J, snes_problem_alpha.A)
            problem_alpha.setTolerances(**self.solver_tols)

            # Set the SNES
            problem_alpha.setType("vinewtonrsls")
            opts.setValue("snes_linesearch_type", "basic")

            # Set the KSP
            # problem_alpha.getKSP().setType("gmres")
            problem_alpha.getKSP().setType("preonly")
            problem_alpha.getKSP().getPC().setType("cholesky")
            problem_alpha.getKSP().getPC().setFactorSolverType("cholmod")

            # problem_alpha.getKSP().setTolerances(atol=1e-9, rtol=1e-9)
            # problem_alpha.getKSP().getPC().setType("mg")
            # problem_alpha.getKSP().getPC().setMGLevels(1)
            # problem_alpha.getKSP().setInitialGuessNonzero(True)

        elif self.petsc_solver == "tao":
            # Define the crack phase problem
            tao_problem_alpha = TAOProblem(
                energy, E_alpha, E_alpha_alpha, alpha, bcs_alpha
            )

            # Set up optimization problem
            problem_alpha = PETSc.TAO().create(comm=MPI.COMM_WORLD)
            problem_alpha.setObjective(tao_problem_alpha.f)
            problem_alpha.setGradient(tao_problem_alpha.F, tao_problem_alpha.b)
            problem_alpha.setHessian(tao_problem_alpha.J, tao_problem_alpha.A)
            problem_alpha.setTolerances(**self.solver_tols)
            opts.setValue("tao_trust0", 0.1)
            opts.setValue("tao_max_it", 10_000)

            # Set up the solver
            # NOTE: Semi working solver
            problem_alpha.setType("bntl")
            opts.setValue("tao_trust0", 0.1)

            # ls = problem_alpha.getLineSearch()
            # ls.setType("unit")
            # # opts.setValue("tao_ls_stepinit", 0.1)
            # ls.setFromOptions()

            # problem_alpha.setType("bnls")
            # problem_alpha.setType("bntl")
            # problem_alpha.setType("bqnls")

            # Set the tolerances
            # opts.setValue("tao_gatol", 1e-6)  # ||g(X)||
            # opts.setValue("tao_grtol", 1e-4)  # ||g(X)|| / |f(X)|
            # opts.setValue("tao_gttol", 1e-4)  # ||g(X)|| / ||g(X0)||
            # opts.setValue("tao_max_it", 10_000)
            # problem_alpha.setType("bncg")
            # opts.setValue("tao_bncg_type", "ssml_bfgs")

        # Define lower and upper bounds functions for the crack phase field
        self.alpha_lb = alpha.copy()
        self.alpha_ub = alpha.copy()
        # Set the upper bound
        with self.alpha_ub.x.petsc_vec.localForm() as alpha_ub_local:
            alpha_ub_local.set(1.0)
        fem.set_bc(self.alpha_ub.x.petsc_vec, bcs_alpha)
        # Set the crack phrase boundary bound (Note: they are passed as reference and not as values)
        problem_alpha.setVariableBounds(
            self.alpha_lb.x.petsc_vec, self.alpha_ub.x.petsc_vec
        )

        # Set options
        problem_alpha.setFromOptions()

        # Display information about the displacement solver
        problem_alpha.view()
        # Store the problem on alpha in subproblems
        self.problem_alpha = problem_alpha

    def define_initial_crack_field(self, pars, domain, state, model, bcs_alpha):
        """Define the initial crack field.

        This method computes the initial crack field based on the provided initial crack configuration
        and assigns it to the crack phase variable `alpha`.

        Parameters
        ----------
        pars : dict
            Dictionary containing parameters for configuring the initial crack field.
        domain : Domain
            The domain object representing the computational domain.
        state : dict
            Dictionary containing state variables.
        model : fragma.models.BaseModel
            The material model used in the simulation.
        bcs_alpha: List
            List of boundary conditions for the crack phase sub-problem.
        """
        # Get the crack phase
        alpha = state["alpha"]
        # Get the cracks
        initial_cracks = pars.get("initial_crack", [])

        # Define the initial crack field
        def alpha_init(x):
            # Initialize the crack phase field
            a = np.zeros((x.shape[1],))
            for crack in initial_cracks:
                # Get points of the crack
                c1 = np.array(crack["p1"])
                c2 = np.array(crack["p2"])
                # Get the width of the crack
                w = crack["width"]
                # Compute the curvilinear abscissa of the orthogonal projection of x on the crack
                c1_x = (x - c1[:, np.newaxis]).transpose()
                t1 = np.dot(c1_x, c2 - c1) / np.linalg.norm(c2 - c1) ** 2
                # Clip t
                t = np.clip(t1, 0, 1)
                # Compute the position of the orthogonal projection
                x_c = np.outer(c1, 1 - t) + np.outer(c2, t)
                # Compute the distance to the current crack
                d = np.linalg.norm(x - x_c, axis=0)
                # Compute the crack contribution to the crack field
                a_new = np.where(d < w, 1, 0)
                # Update the crack field
                a = np.maximum(a, a_new)
            return a

        # Interpolate the initial crack field onto alpha
        alpha.interpolate(alpha_init)

        # Add the boundary conditions
        fem.set_bc(alpha.x.petsc_vec, bcs_alpha)

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
        # Initialize the crack bcs
        bcs_alpha = []
        # Get the dimensions of domain and facets
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
        if "crack" in boundaries:
            bcs_alpha.append(fem.dirichletbc(1.0, boundaries["crack"], V_alpha))
        # Create the uncrackable crack boundary condition
        bcs_alpha_noncrackable = [
            fem.dirichletbc(0.0, b_dof, V_alpha)
            for boundary, b_dof in boundaries.items()
            if boundary.startswith("non-crackable")
        ]
        bcs_alpha += bcs_alpha_noncrackable
        return bcs_alpha

    def update_boundary_conditions(self, l: float):
        """
        Update boundary conditions for the crack phase sub-problem.

        This method updates the crack phase boundary conditions based on the current time.

        Parameters
        ----------
        l : float
            Load factor.
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
        # if self.petsc_solver == "tao" and t < 1:
        #     self.problem_alpha.setTolerances(gatol=0.1, grtol=1e-5, gttol=1e-5)
        # elif self.petsc_solver == "tao" and t >= 1:
        #     self.problem_alpha.setTolerances(gatol=1e-6, grtol=1e-4, gttol=1e-4)

        # Update the crack phase lower bound
        self.alpha.x.petsc_vec.copy(self.alpha_lb.x.petsc_vec)
        # Update of boundary conditions ?
        self.update_boundary_conditions(t)

    def solve(self):
        """Solve the crack phase sub-problem."""
        # Initialiaze the restart in case of solver failure
        reason = PETSc.SNES.ConvergedReason.DIVERGED_TR_DELTA
        restart_reasons = [
            PETSc.SNES.ConvergedReason.DIVERGED_TR_DELTA,
            PETSc.SNES.ConvergedReason.DIVERGED_DTOL,
            PETSc.SNES.ConvergedReason.DIVERGED_LINE_SEARCH,
            PETSc.SNES.ConvergedReason.DIVERGED_LINEAR_SOLVE,
            PETSc.SNES.ConvergedReason.DIVERGED_MAX_IT,
            PETSc.SNES.ConvergedReason.DIVERGED_FNORM_NAN,
            PETSc.TAO.ConvergedReason.DIVERGED_TR_REDUCTION,
        ]
        # Count the number of restart
        max_restart = 10
        restart_counter = 0
        # While loop to restart on failure
        while reason in restart_reasons and restart_counter <= max_restart:
            if self.petsc_solver == "snes":
                self.problem_alpha.solve(None, self.alpha.x.petsc_vec)
            elif self.petsc_solver == "tao":
                self.problem_alpha.solve(self.alpha.x.petsc_vec)
            self.alpha.x.scatter_forward()
            # Check if the solver converged
            reason = self.problem_alpha.getConvergedReason()
            if reason in restart_reasons:
                print("Restarting the crack phase solver due to failure to converge.")
                # Increment the restart counter
                restart_counter += 1

        # Raise an error if the maximum number of restart is reached
        if restart_counter >= max_restart:
            raise ValueError("Max number of restart of the nonlinear solver reached!")
