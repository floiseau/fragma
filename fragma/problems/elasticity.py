from dolfinx import fem
import ufl

from problems.base_problem import BaseProblem
from models import ElasticModel
from subproblems import DisplacementSubProblem


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

    def define_subproblems(self):
        # Define the displacement problem
        self.subproblems["u"] = DisplacementSubProblem(
            self.pars, self.domain, self.state, self.model
        )

    def solve_iteration(self):
        # Solve the displacement problem
        self.subproblems["u"].solve()
