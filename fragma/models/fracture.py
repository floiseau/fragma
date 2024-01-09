import time

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, default_scalar_type, la
from dolfinx.fem.petsc import LinearProblem
import ufl

from utils.snes_problem import SNESProblem
from models.solver import Solver


class FractureSolver(Solver):
    """TODO"""

    def __init__(self, pars):
        super().__init__(pars)

    def define_state_variables(self):
        ### Variational formulation
        print("\n████ DEFINITION OF THE STATE VARIABLES")
        # Define the displacement field
        element_u = ufl.VectorElement("Lagrange", self.domain.ufl_cell(), degree=1)
        self.V_u = fem.FunctionSpace(self.domain, element_u)
        u = fem.Function(self.V_u, name="Displacement")
        # Define the fracture phase field
        element_alpha = ufl.FiniteElement("Lagrange", self.domain.ufl_cell(), degree=1)
        self.V_alpha = fem.FunctionSpace(self.domain, element_alpha)
        alpha = fem.Function(self.V_alpha, name="CrackPhase")
        # Define the state vector
        self.state = {"u": u, "alpha": alpha}

    def eps(self, u):
        return ufl.sym(ufl.grad(u))

    def a(self, alpha):
        # Residual crack phase
        alpha_res = self.pars["numerical"]["alpha_res"]
        # Get the model
        deg_model = self.pars["model"]["model"]
        # Compute a
        match deg_model:
            case "AT1":
                return (1 - alpha) ** 2 + alpha_res
            case "AT2":
                return (1 - alpha) ** 2 + alpha_res
            case _:
                raise ValueError(
                    f"The degradation model named '{deg_model}' does not exists."
                )

    def w(self, alpha):
        # Get the model
        deg_model = self.pars["model"]["model"]
        # Compute w
        match deg_model:
            case "AT1":
                return alpha
            case "AT2":
                return alpha**2
            case _:
                raise ValueError(
                    f"The degradation model named '{deg_model}' does not exists."
                )

    def cw(self):
        deg_model = self.pars["model"]["model"]
        match deg_model:
            case "AT1":
                return 8 / 3
            case "AT2":
                return 2
            case _:
                raise ValueError(
                    f"The degradation model named '{deg_model}' does not exists."
                )

    def sig(self, u, alpha):
        # Get the elastic parameters
        E = self.pars["mechanical"]["E"]
        nu = self.pars["mechanical"]["nu"]
        # Compute Lame coefficient
        la = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        # Check the 2D assumption
        if self.pars["model"]["dim"] == 2:
            assumption = self.pars["model"]["2D_assumption"]
            match assumption:
                case "plane_stress":
                    print("Plane stress assumption")
                    la = 2 * mu * la / (la + 2 * mu)
                case "plane_strain":
                    print("Plane strain assumption")
                case _:
                    raise ValueError(f"The 2D assumption '{assumption}' in unknown")
        # Compute the stess
        return self.a(alpha) * (
            la * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0 * mu * self.eps(u)
        )

    def define_total_energy(self):
        # Get the dimension of the domain
        dim = self.domain.topology.dim
        # Get the integrands
        self.dx = ufl.Measure("dx", domain=self.domain)
        ds = ufl.Measure("ds", domain=self.domain)
        # Define the imposed stress on the remaining of the boundary
        T = fem.Constant(self.domain, default_scalar_type([0 for d in range(dim)]))
        # Define the volumic forces
        f = fem.Constant(self.domain, default_scalar_type([0 for d in range(dim)]))
        # Get state variables
        u = self.state["u"]
        alpha = self.state["alpha"]
        # Get the parameters
        Gc = self.pars["mechanical"]["Gc"]
        ell = self.pars["mechanical"]["ell"]
        cw = self.cw()
        # Define the energy terms
        elastic_energy = 0.5 * ufl.inner(self.sig(u, alpha), self.eps(u)) * self.dx
        dissipated_energy = (
            Gc
            / cw
            * (self.w(alpha) / ell + ell * ufl.dot(ufl.grad(alpha), ufl.grad(alpha)))
            * self.dx
        )
        external_work = ufl.dot(f, u) * self.dx + ufl.dot(T, u) * ds
        # Define the total energy
        self.total_energy = elastic_energy + dissipated_energy - external_work

    def define_displacement_problem(self):
        # Define the boundary condition functions for displacement
        self.define_displacement_boundary_condition_functions()
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
            bcs=self.bcs_u,
            u=u,
            petsc_options={
                "ksp_type": "cg",
                "ksp_rtol": 1e-8,
                "ksp_atol": 1e-10,
                "ksp_max_it": 1000,
                "pc_type": "gamg",
                "pc_gamg_agg_nsmooths": 1,  #  "pc_gamg_esteig_ksp_type": "cg",
            },
            # petsc_options={"ksp_type": "cg", "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000, "pc_type":"hypre"},
            # petsc_options={"ksp_type": "preonly", "pc_type":"lu", "pc_factor_solver_type": "mumps"},
        )
        # Display information about the displacement solver
        self.problem_u.solver.view()

    def define_crack_phase_boundary_condition_functions(self):
        # Add the damage boundary conditions if there is an initial crack
        if "crack" in self.pars["mesh"]["physical_groups"]:
            # Get the physical groups (mapping between pg and their indices)
            facets_tags_values = self.pars["mesh"]["physical_groups"]
            # Get the facets indices
            boundary_facets = {}
            for facet_name, facet_value in facets_tags_values.items():
                boundary_facets[facet_name] = self.facet_tags.indices[
                    self.facet_tags.values == facet_value
                ]
            # Get the dimensions of domain and facets
            dim = self.domain.topology.dim
            fdim = self.domain.topology.dim - 1
            # Get boundary dofs (per comp)
            boundaries = {
                f"{facet_name}": fem.locate_dofs_topological(
                    self.V_alpha,
                    fdim,
                    boundary_facet,
                )
                for facet_name, boundary_facet in boundary_facets.items()
            }
            # Create the crack boundary condition
            bcs_alpha_crack = [fem.dirichletbc(1.0, boundaries["crack"], self.V_alpha)]
            # Create the uncrackable crack boundary condition
            bcs_alpha_noncrackable = [
                fem.dirichletbc(0.0, b_dof, self.V_alpha)
                for boundary, b_dof in boundaries.items()
                if boundary.startswith("non-crackable")
            ]
            #
            self.bcs_alpha = bcs_alpha_crack + bcs_alpha_noncrackable
        else:
            self.bcs_alpha = []

    def define_crack_phase_problem(self):
        print("\n████ DEFINITION OF THE CRACK PHASE PROBLEM")
        # Define the boundary conditions
        self.define_crack_phase_boundary_condition_functions()
        # Get the state variables
        u, alpha = self.state["u"], self.state["alpha"]
        # Derivative of the energy with respect to crack phase
        E_alpha = ufl.derivative(
            self.total_energy, alpha, ufl.TestFunction(self.V_alpha)
        )
        E_alpha_alpha = ufl.derivative(E_alpha, alpha, ufl.TrialFunction(self.V_alpha))

        # Define the crack phase problem
        snes_problem_alpha = SNESProblem(E_alpha, E_alpha_alpha, alpha, self.bcs_alpha)
        # Initialize the LHS
        b = fem.petsc.create_vector(snes_problem_alpha.L)
        # Initialize the jacobian
        J = fem.petsc.create_matrix(fem.form(snes_problem_alpha.a))

        # Create Newton solver
        self.problem_alpha = PETSc.SNES().create()
        self.problem_alpha.setType("vinewtonrsls")
        self.problem_alpha.setFunction(snes_problem_alpha.F, b)
        self.problem_alpha.setJacobian(snes_problem_alpha.J, J)
        self.problem_alpha.setTolerances(rtol=1.0e-9, max_it=50)

        self.problem_alpha.getKSP().setType("preonly")
        self.problem_alpha.getKSP().setTolerances(rtol=1.0e-9)
        self.problem_alpha.getKSP().getPC().setType("lu")
        self.problem_alpha.getKSP().getPC().setFactorSolverType("mumps")

        # Define lower and upper bounds functions for the crack phase field
        self.alpha_lb = fem.Function(self.V_alpha, name="Lower bound")
        self.alpha_ub = fem.Function(self.V_alpha, name="Upper bound")
        # Set the lower bound
        with self.alpha_lb.vector.localForm() as alpha_lb_local:
            alpha_lb_local.set(0.0)
        fem.set_bc(self.alpha_lb.vector, self.bcs_alpha)
        # Set the upper bound
        one_alpha = fem.Function(self.V_alpha)
        with self.alpha_ub.vector.localForm() as alpha_ub_local:
            alpha_ub_local.set(1.0)
        fem.set_bc(self.alpha_ub.vector, self.bcs_alpha)
        # TODO Remove this
        # Add the bounds to state in order to export them
        self.state["alpha_lb"] = self.alpha_lb
        self.state["alpha_ub"] = self.alpha_ub
        # Set the crack phrase boundary bound (Note: they are passed as reference and not as values)
        self.problem_alpha.setVariableBounds(self.alpha_lb.vector, self.alpha_ub.vector)

        # Initialise alpha
        self.alpha_lb.vector.copy(alpha.vector)

        # Display information about the displacement solver
        self.problem_alpha.view()

    def define_problems(self):
        # Define the displacement problem
        self.define_displacement_problem()
        # Define the crack phase problem
        self.define_crack_phase_problem()

    def monitor(self, t, error_L2, time_u, time_alpha):
        if MPI.COMM_WORLD.rank == 0:
            print(
                f"Iteration: {t:3d}, Error: {error_L2:3.4e}, Time u: {time_u:3.4e}s, Time alpha: {time_alpha:3.4e}s"
            )

    def solve_iteration(self):
        # Get the state
        u, alpha = self.state["u"], self.state["alpha"]
        # Update the crack phase lower bound
        alpha.vector.copy(self.alpha_lb.vector)
        # Define alpha at previous iteration for irreversibility
        alpha_old = fem.Function(alpha.function_space)
        with alpha_old.vector.localForm() as alpha_old_local:
            alpha_old_local.set(0.0)
        # Perform the alternate minimization
        for t in range(self.pars["numerical"]["max_iter"]):
            # Solve the displacement problem
            time_u_start = time.perf_counter()
            self.problem_u.solve()
            time_u = time.perf_counter() - time_u_start
            # Solve the crack phase problem
            time_alpha_start = time.perf_counter()
            self.problem_alpha.solve(None, alpha.vector)
            time_alpha = time.perf_counter() - time_alpha_start
            # Check error
            L2_error = fem.form(
                ufl.inner(alpha - alpha_old, alpha - alpha_old) * self.dx
            )
            error_L2 = np.sqrt(fem.assemble_scalar(L2_error))
            # Update alpha_old
            alpha.vector.copy(alpha_old.vector)
            # Display information
            self.monitor(t, error_L2, time_u, time_alpha)
            # Check convergence
            if error_L2 <= self.pars["numerical"]["atol"]:
                break
        else:
            raise RuntimeError(
                f"Could not converge after {t:3d} iteration, error {error_L2:3.4e}"
            )
