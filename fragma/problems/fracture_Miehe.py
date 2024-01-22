import time

from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl

from models import FractureModelMiehe
from problems.base_problem import BaseProblem
from subproblems import DisplacementSubProblem, CrackPhaseSubProblemMiehe
from utils.build_nullspace import build_elasticity_nullspace


class FractureProblemMiehe(BaseProblem):
    """TODO"""

    def __init__(self, pars):
        # Create the elasticity model
        self.model = FractureModelMiehe(pars)
        # Initialise parent class
        super().__init__(pars)

    def define_state_variables(self):
        ### Variational formulation
        print("\n████ DEFINITION OF THE STATE VARIABLES")
        # Define the history field
        element_H = ufl.FiniteElement("DG", self.domain.mesh.ufl_cell(), degree=0)
        self.V_H = fem.FunctionSpace(self.domain.mesh, element_H)
        H = fem.Function(self.V_H, name="History")
        # Define the displacement field
        element_u = ufl.VectorElement("Lagrange", self.domain.mesh.ufl_cell(), degree=1)
        self.V_u = fem.FunctionSpace(self.domain.mesh, element_u)
        u = fem.Function(self.V_u, name="Displacement")
        # Define the fracture phase field
        element_alpha = ufl.FiniteElement(
            "Lagrange", self.domain.mesh.ufl_cell(), degree=1
        )
        self.V_alpha = fem.FunctionSpace(self.domain.mesh, element_alpha)
        alpha = fem.Function(self.V_alpha, name="CrackPhase")
        # Define the state vector
        self.state = {"H": H, "u": u, "alpha": alpha}

    def define_subproblems(self):
        # Define the displacement problem
        self.subproblems["u"] = DisplacementSubProblem(
            self.pars, self.domain, self.state, self.model
        )
        # Define the displacement problem
        self.subproblems["alpha"] = CrackPhaseSubProblemMiehe(
            self.pars, self.domain, self.state, self.model
        )

    def monitor(self, time_u, time_alpha):
        if MPI.COMM_WORLD.rank == 0:
            print(
                f"Iteration: 1, Time u: {time_u:3.4e}s, Time alpha: {time_alpha:3.4e}s"
            )

    def solve_iteration(self):
        # Get the state
        u, alpha, H = self.state["u"], self.state["alpha"], self.state["H"]
        # 1 - Update the history field
        # Get the function space for Phi0
        V_H = H.function_space
        # Compute Phi0 (symbolic, UFL)
        Phi0_ufl = ufl.inner(self.model.sig(self.state), self.model.eps(self.state))
        # Generate the FEM expression
        Phi0_expr = fem.Expression(Phi0_ufl, V_H.element.interpolation_points())
        # Generate the function and interpolate it
        Phi0 = fem.Function(V_H)
        Phi0.interpolate(Phi0_expr)
        # Compute H
        H.vector[:] = np.maximum(Phi0.vector[:], H.vector[:])
        H.vector.assemble()
        # 2 - Solve the crack phase problem
        time_alpha_start = time.perf_counter()
        self.subproblems["alpha"].solve()
        time_alpha = time.perf_counter() - time_alpha_start
        # 3 - Solve the displacement problem
        time_u_start = time.perf_counter()
        self.subproblems["u"].solve()
        time_u = time.perf_counter() - time_u_start
        # Display information
        self.monitor(time_u, time_alpha)
