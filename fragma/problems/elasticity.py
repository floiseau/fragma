from petsc4py import PETSc

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl

from problems.base_problem import BaseProblem
from models import ElasticModel
from utils.build_nullspace import build_elasticity_nullspace


class ElasticityProblem(BaseProblem):
    """
    Solver for 2D elasticity problem (in plane strain or plain stress).
    The loading are proportional to time.
    """

    def __init__(self, pars):
        # Create the elasticity model
        self.model = ElasticModel(pars)
        # Initialise parent class
        super().__init__(pars)

    def define_state_variables(self):
        ### Variational formulation
        print("\n████ DEFINITION OF THE STATE VARIABLES")
        # Define the elements
        element_u = ufl.VectorElement("Lagrange", self.domain.mesh.ufl_cell(), 1)
        # Define finite element spaces
        self.V_u = fem.FunctionSpace(self.domain.mesh, element_u)
        # Define the state variables
        u = fem.Function(self.V_u, name="Displacement")
        # Define the state vector
        self.state = {"u": u}

    def define_displacement_problem(self):
        print("\n████ DEFINITION OF THE DISPLACEMENT PROBLEM")
        # Define the boundary condition functions for displacement
        self.define_displacement_boundary_condition_functions()
        # Get the state variables
        u = self.state["u"]
        # Get the total energy from the model
        energy = self.model.energy(self.state, self.domain.mesh)
        # Derivative of the energy with respect to displacement to obtain the linear problem to determine the stationary point
        E_u = ufl.derivative(energy, u, ufl.TestFunction(self.V_u))
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
