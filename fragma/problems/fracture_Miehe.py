import time

from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

from dolfinx import fem, default_scalar_type
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
        # Get the dimension
        dim = self.domain.mesh.geometry.dim
        # Define the history field
        element_H = ufl.FiniteElement("DG", self.domain.mesh.ufl_cell(), degree=0)
        self.V_H = fem.FunctionSpace(self.domain.mesh, element_H)
        H = fem.Function(self.V_H, name="History")
        H.interpolate(lambda x: default_scalar_type(0.0) * x[0])
        # Define the displacement field
        element_u = ufl.VectorElement("Lagrange", self.domain.mesh.ufl_cell(), degree=1)
        self.V_u = fem.FunctionSpace(self.domain.mesh, element_u)
        u = fem.Function(self.V_u, name="Displacement")
        u.interpolate(lambda x: np.array([default_scalar_type(0.0) * x[0]] * dim))
        # Define the fracture phase field
        element_alpha = ufl.FiniteElement(
            "Lagrange", self.domain.mesh.ufl_cell(), degree=1
        )
        self.V_alpha = fem.FunctionSpace(self.domain.mesh, element_alpha)
        alpha = fem.Function(self.V_alpha, name="CrackPhase")
        alpha.interpolate(lambda x: default_scalar_type(0.0) * x[0])
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
        # TODO If dt changes, update constant for viscosity

        # 1 - Update the history field
        self.model.update_history(self.state)
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

    # def solve(self):
    #     print("\n████ RESOLUTION")
    #     # Start the loading iterations
    #     t_max = self.pars["loading"]["t_max"]
    #     # Get the state
    #     u, alpha, H = self.state["u"], self.state["alpha"], self.state["H"]

    #     # Define Phi0
    #     V_H = H.function_space
    #
    #     # Compute Phi0 (symbolic, UFL)
    #     Phi0_ufl = ufl.inner(self.model.sig(self.state), self.model.eps(self.state))
    #     # Generate the FEM expression
    #     Phi0_expr = fem.Expression(Phi0_ufl, V_H.element.interpolation_points())
    #     # Generate the function and interpolate it
    #     Phi0 = fem.Function(V_H)
    #     Phi0.interpolate(Phi0_expr)

    #     # Define old state variables values
    #     u_old = u.copy()
    #     alpha_old = alpha.copy()
    #     H_old = H.copy()
    #     Phi0_old = Phi0.copy()

    #     # Iterate through time
    #     t = 0
    #     dt = 1
    #     tol = 1e-1
    #     while t <= t_max:
    #         # Display information
    #         print(f"== Time {t}/{t_max}")

    #         # 1 - Update the history field
    #         Phi0.interpolate(Phi0_expr)
    #         H.vector[:] = np.maximum(Phi0.vector[:], H.vector[:])
    #         H.vector.assemble()

    #         # Update the old values
    #         u_old.vector[:] = u.vector
    #         alpha_old.vector[:] = alpha.vector
    #         H_old.vector[:] = H.vector
    #         Phi0_old.vector[:] = Phi0.vector

    #         # Perform the load step adaptation
    #         converged = False
    #         while not converged:

    #             ### Solve the full step
    #             # Update subproblems
    #             self.update_subproblems(t+dt)
    #             # 2 - Solve the crack phase problem
    #             self.subproblems["alpha"].solve()
    #             alpha_full = alpha.copy()
    #             # 3 - Solve the displacement problem
    #             self.subproblems["u"].solve()

    #             ### Reset
    #             # Update the old values
    #             u.vector[:] = u_old.vector
    #             u.vector.assemble()
    #             alpha.vector[:] = alpha_old.vector
    #             alpha.vector.assemble()
    #             H.vector[:] = H_old.vector
    #             H.vector.assemble()
    #             Phi0.vector[:] = Phi0_old.vector
    #             Phi0.vector.assemble()

    #             ### Solve the two half steps
    #             # Update subproblems
    #             self.update_subproblems(t+dt/2)
    #             # 2a - Solve the crack phase problem
    #             self.subproblems["alpha"].solve()
    #             # 3a - Solve the displacement problem
    #             self.subproblems["u"].solve()
    #             # Update subproblems
    #             self.update_subproblems(t+dt)
    #             # 1b - Update the history field
    #             Phi0.interpolate(Phi0_expr)
    #             H.vector[:] = np.maximum(Phi0.vector[:], H.vector[:])
    #             H.vector.assemble()
    #             # 2a - Solve the crack phase problem
    #             self.subproblems["alpha"].solve()
    #             alpha_half = alpha.copy()
    #             # 3a - Solve the displacement problem
    #             self.subproblems["u"].solve()

    #             # Compute the error
    #             tau = np.max(np.abs(alpha_full.vector[:] - alpha_half.vector[:]))
    #             # Check the convergence
    #             converged = tau < tol
    #             # Increment time if converged
    #             if converged:
    #                 t += dt
    #             # Compute the new time step
    #             new_dt = np.sqrt(tol/(2*tau))
    #             dt = 0.9*dt*min(max(new_dt, 0.3), 2)
    #             print(f"dt = {dt}")
    #         # Apply post processing
    #         self.postprocessor.postprocess()
    #         # Export the results
    #         self.exporter.export(t)
    #     # End export
    #     self.exporter.end()
