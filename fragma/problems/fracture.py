"""
Fracture Problem Module
=======================

This module provides a solver for fracture problems using a phase-field model.

Classes:
    FractureProblem: Solver for fracture problems using a phase-field model.
"""

import time

from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl

from models import FractureModel
from problems.base_problem import BaseProblem
from subproblems import create_displacement_subproblem, CrackPhaseSubProblem
from utils.build_nullspace import build_elasticity_nullspace


class FractureProblem(BaseProblem):
    """
    Solver for fracture problems using a phase-field model.

    This class inherits from BaseProblem and provides functionality to solve
    fracture problems using a phase-field model. It defines state variables,
    subproblems, and methods to solve the problem over time.

    Attributes
    ----------
    model : FractureModel
        The fracture model used for solving the problem.
    V_u : dolfinx.FunctionSpace
        Function space for the displacement field.
    V_alpha : dolfinx.FunctionSpace
        Function space for the fracture phase field.
    state : dict
        Dictionary containing the state variables.
    subproblems : dict
        Dictionary containing subproblems of the main problem.
    """

    def __init__(self, pars):
        """
        Initialize the FractureProblem solver.

        Parameters
        ----------
        pars : dict
            Dictionary containing parameters for the problem.
        """
        # Create the elasticity model
        self.model = FractureModel(pars)
        # Initialise parent class
        super().__init__(pars)

    def define_state_variables(self):
        """
        Define the state variables for the fracture problem.

        This method defines the displacement and fracture phase field variables.
        """
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
        """
        Define the subproblems for the fracture problem.

        This method defines the displacement and fracture phase subproblems.
        """
        # Define the displacement problem
        self.subproblems["u"] = create_displacement_subproblem(
            self.pars, self.domain, self.state, self.model
        )
        # Define the displacement problem
        self.subproblems["alpha"] = CrackPhaseSubProblem(
            self.pars, self.domain, self.state, self.model
        )

    def monitor(self, k, l, error_u, error_a, time_u, time_alpha):
        """
        Monitor the progress of the fracture problem solver.

        This method prints information about the current iteration, including
        the iteration number, error, and computation times for solving the
        displacement and fracture phase subproblems.

        Parameters
        ----------
        k : int
            Alternate minimization iteration number.
        l : float
            Load factor.
        error_u : float
            Displacement error between two successive alternate minimization iterations.
        error_a : float
            Crack phase error between two successive alternate minimization iterations.
        time_u : float
            Computation time for solving the displacement subproblem.
        time_alpha : float
            Computation time for solving the fracture phase subproblem.
        """
        if MPI.COMM_WORLD.rank == 0:
            print(
                f"Iteration: {k:3d}, Load factor: {l:3.4e}, Error u: {error_u:3.4e}, Error a: {error_a:3.4e}, Time u: {time_u:3.4e}s, Time alpha: {time_alpha:3.4e}s"
            )

    def solve_iteration(self):
        """
        Solve a single iteration of the fracture problem.

        This method iteratively solves the displacement and fracture phase
        subproblems until convergence is achieved or the maximum number of
        iterations is reached.
        """
        # Get the state
        u, alpha = self.state["u"], self.state["alpha"]
        # Define state at previous iteration for error computation
        u_old, alpha_old = u.copy(), alpha.copy()
        # Initialize the errors
        error_u, error_a = 0, 0
        # Get previous displacement (for over-relaxation)
        relaxation = "omega" in self.pars["numerical"]
        if relaxation:
            omega = self.pars["numerical"]["omega"]
        # Perform the alternate minimization
        for k in range(self.pars["numerical"]["max_iter"]):
            # Solve the displacement problem
            time_u_start = time.perf_counter()
            self.subproblems["u"].solve()
            time_u = time.perf_counter() - time_u_start
            # Perform displacement relaxiation
            if relaxation:
                u.x.array[:] = u_old.x.array + omega * (u.x.array[:] - u_old.x.array[:])
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
            # Check errors (L2)
            error_u = np.max(np.abs(u.x.array - u_old.x.array))
            error_a = np.max(np.abs(alpha.x.array - alpha_old.x.array))
            # Check convergence
            converged = (
                error_u <= self.pars["numerical"]["utol"]
                and error_a <= self.pars["numerical"]["atol"]
            )
            # Display information
            self.monitor(k, self.subproblems["u"].l, error_u, error_a, time_u, time_alpha)
            # Update old fields
            u_old.x.array[:] = u.x.array
            u_old.x.scatter_forward()
            alpha_old.x.array[:] = alpha.x.array
            alpha_old.x.scatter_forward()
            # Stop to iterate if the calculation is converged
            if converged:
                break
        else:
            raise RuntimeError(
                f"Could not converge after {k:3d} iteration, error {error:3.4e}"
            )
