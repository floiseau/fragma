import numpy as np

from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

from models.solver import Solver
from utils.build_nullspace import build_elasticity_nullspace


class ElasticitySolver(Solver):
    """
    Solver for 2D elasticity problem (in plane strain or plain stress).
    The loading are proportional to time.
    """

    def __init__(self, pars):
        super().__init__(pars)

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
        if self.pars["model"]["dim"] == 2:
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
        # Get the dimension of the domain
        dim = self.domain.topology.dim
        # Get the integrands
        dx = ufl.Measure("dx", domain=self.domain)
        ds = ufl.Measure("ds", domain=self.domain)
        # Define the imposed stress on the remaining of the boundary
        T = fem.Constant(self.domain, default_scalar_type([0 for d in range(dim)]))
        # Define the volumic forces
        f = fem.Constant(self.domain, default_scalar_type([0 for d in range(dim)]))
        # Get state variables
        u = self.state["u"]
        # Define the energy terms
        elastic_energy = 0.5 * ufl.inner(self.sig(u), self.eps(u)) * dx
        dissipated_energy = 0.0 * dx
        external_work = ufl.dot(f, u) * dx + ufl.dot(T, u) * ds
        # Define the total energy
        self.total_energy = elastic_energy + dissipated_energy - external_work

    def define_displacement_problem(self):
        print("\n████ DEFINITION OF THE DISPLACEMENT PROBLEM")
        # Define the boundary condition functions for displacement
        self.define_displacement_boundary_condition_functions()
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
        ns = build_elasticity_nullspace(self.V_u)
        self.problem_u.A.setNearNullSpace(ns)
        self.problem_u.A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore
        # Display information about the displacement solver
        self.problem_u.solver.view()

    def define_problems(self):
        # Define the displacement problem
        self.define_displacement_problem()

    def solve_iteration(self):
        # Solve the displacement problem
        self.problem_u.solve()
