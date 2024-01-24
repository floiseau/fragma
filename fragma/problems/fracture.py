import time

from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl

from models import FractureModel
from problems.base_problem import BaseProblem
from subproblems import DisplacementSubProblem, CrackPhaseSubProblem
from utils.build_nullspace import build_elasticity_nullspace


class FractureProblem(BaseProblem):
    """TODO"""

    def __init__(self, pars):
        # Create the elasticity model
        self.model = FractureModel(pars)
        # Initialise parent class
        super().__init__(pars)

    def define_state_variables(self):
        ### Variational formulation
        print("\n████ DEFINITION OF THE STATE VARIABLES")
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
        self.state = {"u": u, "alpha": alpha}

    def define_subproblems(self):
        # Define the displacement problem
        self.subproblems["u"] = DisplacementSubProblem(
            self.pars, self.domain, self.state, self.model
        )
        # Define the displacement problem
        self.subproblems["alpha"] = CrackPhaseSubProblem(
            self.pars, self.domain, self.state, self.model
        )

    def monitor(self, t, error, time_u, time_alpha):
        if MPI.COMM_WORLD.rank == 0:
            print(
                f"Iteration: {t:3d}, Error: {error:3.4e}, Time u: {time_u:3.4e}s, Time alpha: {time_alpha:3.4e}s"
            )

    def solve_iteration(self):
        # Get the state
        u, alpha = self.state["u"], self.state["alpha"]
        # Define alpha at previous iteration for error computation
        alpha_old = fem.Function(alpha.function_space)
        alpha.vector.copy(alpha_old.vector)
        # Get previous displacement (for over-relaxation)
        relaxation = "omega" in self.pars["numerical"]
        if relaxation:
            omega = self.pars["numerical"]["omega"]
            u_old = u.copy()
        # Perform the alternate minimization
        for t in range(self.pars["numerical"]["max_iter"]):
            # Solve the displacement problem
            time_u_start = time.perf_counter()
            self.subproblems["u"].solve()
            time_u = time.perf_counter() - time_u_start
            # Perform displacement relaxiation
            if relaxation:
                u.vector[:] = u_old.vector[:] + omega * (u.vector[:] - u_old.vector[:])
                u.vector.assemble()
            # Solve the crack phase problem
            time_alpha_start = time.perf_counter()
            self.subproblems["alpha"].solve()
            time_alpha = time.perf_counter() - time_alpha_start
            # Perform crack phase relaxation
            if relaxation:
                dalpha = alpha.vector[:] - alpha_old.vector[:]
                omega_bar = omega * np.ones((len(dalpha),))
                new_alpha = alpha_old.vector[:] + omega_bar * dalpha
                # Add a counter
                while new_alpha.max() > 1.0:
                    omega_bar = 1 / 2 * (1 + omega_bar)
                    new_alpha = alpha_old.vector[:] + omega_bar * dalpha
                alpha.vector[:] = alpha_old.vector[:] + omega_bar * dalpha
                alpha.vector.assemble()
                print(f"Crack phase relaxation: f{omega_bar[0]}")
            # Check error (L2)
            error = np.max(alpha.vector[:] - alpha_old.vector[:])
            # Update alpha_old
            alpha.vector.copy(alpha_old.vector)
            # Display information
            self.monitor(t, error, time_u, time_alpha)
            # Check convergence
            if error <= self.pars["numerical"]["atol"]:
                break
        else:
            raise RuntimeError(
                f"Could not converge after {t:3d} iteration, error {error:3.4e}"
            )
