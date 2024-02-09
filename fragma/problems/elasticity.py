"""
Elasticity Problem Solver
=========================

This module provides a solver for 2D elasticity problems (in plane strain or plain stress) and 3D elasticity problems.

Classes:
    ElasticityProblem: Solver for 2D and 3D elasticity problems.

"""

from dolfinx import fem
import ufl

from problems.base_problem import BaseProblem
from models import ElasticModel
from subproblems import create_displacement_subproblem


class ElasticityProblem(BaseProblem):
    """
    Solver for 2D elasticity problem (in plane strain or plain stress).
    The loading are proportional to time.

    Attributes
    ----------
    model : ElasticModel
        The elasticity model used for solving the problem.
    V_u : dolfinx.FunctionSpace
        The function space for the displacement field.
    state : dict
        Dictionary containing the state variables.
    """

    def __init__(self, pars):
        """
        Initialize the ElasticityProblem solver.

        Parameters
        ----------
        pars : dict
            Dictionary containing parameters for the problem.
        """
        # Create the elasticity model
        self.model = ElasticModel(pars)
        # Initialise parent class
        super().__init__(pars)

    def define_state_variables(self):
        """
        Define the state variables for the problem.
        """
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

    def define_subproblems(self):
        """
        Define the subproblems for the elasticity problem.
        """
        print("\n████ DEFINITION OF THE SUB-PROBLEMS")
        # Define the displacement problem
        self.subproblems["u"] = create_displacement_subproblem(
            self.pars, self.domain, self.state, self.model
        )

    def solve_iteration(self):
        """
        Solve a single iteration of the elasticity problem.
        """
        # Solve the displacement problem
        self.subproblems["u"].solve()
