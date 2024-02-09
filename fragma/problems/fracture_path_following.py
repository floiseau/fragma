import time
import re

from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
from scipy.optimize import root_scalar

from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

from models import FractureModelPathFollowing
from problems.base_problem import BaseProblem
from subproblems import DisplacementSubProblemPathFollowing, CrackPhaseSubProblem
from utils.build_nullspace import build_elasticity_nullspace


class FractureProblemPathFollowing(BaseProblem):
    """TODO"""

    def __init__(self, pars):
        # Create the elasticity model
        self.model = FractureModelPathFollowing(pars)
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
        # Define the crack phase value a previous load step
        alpha_old = fem.Function(self.V_alpha, name="CrackPhaseOld")
        # Define the load factor
        Lamb = fem.Constant(self.domain.mesh, default_scalar_type(0.0))
        # Define the crack size
        Gamma = fem.Constant(self.domain.mesh, default_scalar_type(0.0))
        # Define the state vector
        self.state = {
                "u": u, "alpha": alpha,
                "Lamb": Lamb, "Gamma": Gamma, "alpha_old": alpha_old}

    def define_subproblems(self):
        # Define the displacement problem
        self.subproblems["u"] = DisplacementSubProblemPathFollowing(
            self.pars, self.domain, self.state, self.model
        )
        # Define the displacement problem
        self.subproblems["alpha"] = CrackPhaseSubProblem(
            self.pars, self.domain, self.state, self.model
        )

    def monitor(self, err_alpha, lamb, dlamb, Gamma, dGamma):
        if MPI.COMM_WORLD.rank == 0:
            print(f"* Summary of iteration:")
            print(f" | Error phase field            : {err_alpha:.3g}")
            print(f" | Crack length increment       : {dGamma:.3g}")
            print(f" | Final crack length           : {Gamma:.3g}")
            print(f" | New load factor              : {lamb:.3g}")
            print(f" | New load fractor inc         : {dlamb:.3g}")


    def solve():
        print("\n████ RESOLUTION")
        # Get the state
        u, alpha = self.state["u"], self.state["alpha"]
        # Get the path-following constants
        Lamb, Gamma = self.state["Lamb"], self.state["Gamma"]
        # Initialize the load factor
        Lamb.value = 0
        # Initialize time step
        t = 0
        # Initialize the crack size
        print("* Initialize the crack size")
        dx = ufl.Measure("dx", domain=self.domain.mesh)
        Gamma_form = fem.form(alpha * dx)
        Gamma = fem.assemble_scalar(Gamma_form)
        print(f" | Crack size : {Gamma:.3g}")
