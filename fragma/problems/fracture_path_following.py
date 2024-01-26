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


class FractureProblemPathFollowing(BaseProblem):
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

    def monitor(self, err_alpha, lamb, dlamb, Gamma, dGamma):
        if MPI.COMM_WORLD.rank == 0:
            print(f"* Summary of iteration:")
            print(f" | Error phase field            : {err_alpha:.3g}")
            print(f" | New load factor              : {lamb:.3g}")
            print(f" | New load fractor inc         : {dlamb:.3g}")
            print(f" | Final crack length           : {Gamma:.3g}")
            print(f" | Crack length increment       : {dGamma:.3g}")
            print("")

    def solve(self):
        print("\n████ RESOLUTION")
        # Get integrand
        dx = ufl.Measure("dx", domain=self.domain.mesh)
        # Get the state
        u, alpha = self.state["u"], self.state["alpha"]
        # Initialize alpha (diffuse pre-crack)
        self.subproblems["alpha"].solve()
        # Set the crack surface increment
        dGamma = 5e-7
        # Set the tolerance on the crack surface increment
        tol_dGamma = 0.5
        # Set load factor changes bounds
        dlamb_max = 0.02
        # Initialize time step
        t = 0
        # Initialize the load factor
        lamb_k = 0.001
        # Initialize the crack surface
        Gamma_k = fem.assemble_scalar(fem.form(alpha * dx))
        # Define alpha at previous iteration (k) (for error computation)
        alpha_k = fem.Function(alpha.function_space)
        alpha.vector.copy(alpha_k.vector)
        # Initialize time load loop
        end = False
        while not end:  # Time steps loop
            # Display information
            print(f"\n== Time step {t:08d}")
            # Update the crack phase lower bound
            self.subproblems["alpha"].update(t)
            # Store the crack surface at begin of load step
            Gamma_0 = Gamma_k
            # Store the load factor at the begin of load step
            lamb_0 = lamb_k
            # Initialize the new load factor to the one at the begin of the step
            lamb = lamb_0
            # Initialize the constraint
            C_k = None
            # Initialize the iteration
            converged = False
            k = 0
            while not converged:  # Iteration loop
                # Store the load factor from previous iteration (km1)
                lamb_km1 = lamb_k
                # Set the load factor for the current iteration
                lamb_k = lamb
                # Display information
                print(
                    f"--- Iteration {k} with lambda={lamb_k:.3g} (dlambda={lamb_k-lamb_0:.3})"
                )

                # Update boundary conditions with load factor
                self.subproblems["u"].update(lamb_k)

                # Solve the u and alpha problems
                am_converged = False
                k_sub = 0
                print("* Alternate minimization")
                while not am_converged and k_sub < 500:  # Alternate minimization loop
                    # Increment the counter
                    k_sub += 1
                    # Solve the displacement problem
                    self.subproblems["u"].solve()
                    # Store previous alpha
                    alpha.vector.copy(alpha_k.vector)
                    # Solve the crack phase problem
                    self.subproblems["alpha"].solve()
                    # Check error (L2)
                    err_alpha = np.max(alpha.vector[:] - alpha_k.vector[:])
                    # Update alpha_k
                    alpha.vector.copy(alpha_k.vector)
                    # Check convergence
                    am_converged = err_alpha <= 1e-5
                    # Print some information
                    print(
                        f" | Iteration: {k_sub:03d}; Error: {err_alpha:.4e}",
                        end="\r" if not am_converged else "\n",
                    )

                if not am_converged:
                    print(" | Did not converge, retry with a smaller increment.")
                    lamb -= abs(lamb_k - lamb_0) / 2
                    continue

                # Compute the crack surface
                Gamma_km1 = Gamma_k
                Gamma_k = fem.assemble_scalar(fem.form(alpha * dx))
                dGamma_k = Gamma_k - Gamma_0
                # Compute the constraint
                C_km1 = C_k
                C_k = Gamma_k - (Gamma_0 + dGamma)

                # Check crack growth
                crack_growth_too_small = dGamma_k < dGamma * (1 - tol_dGamma)
                crack_growth_too_large = dGamma_k > dGamma * (1 + tol_dGamma)
                crack_growth_ok = (
                    not crack_growth_too_small and not crack_growth_too_large
                )

                print("* Load control:")
                if crack_growth_ok:  # Keep same load factor
                    print(" | Crack growth is OK.")
                    print(" | Keeping the same load factor for next load step.")
                    # The crack growth is small enought
                    converged = True
                else:  # Load factor is too large
                    # Crack growth is too large
                    converged = False
                    if crack_growth_too_small:
                        print(" | Crack growth is too small.")
                    else:
                        print(" | Crack growth is too high.")

                    if k == 0:  # First iteration
                        print(" | First iteration, perturbate the load factor")
                        # Perturbate the load factor
                        lamb += 0.01 * abs(lamb)
                    elif C_k - C_km1 == 0:  # Crack length did not change
                        print(" | No crack growth, increase the load factor")
                        # Increase the load factor (double the current iteration increment)
                        # NOTE This is complicated to manage
                        lamb += 1.1 * abs(lamb)
                    else:
                        print(" | Updating dlamb with secant method")
                        # Compute the new increment of load factor
                        dlamb = -C_k * (lamb_k - lamb_km1) / (C_k - C_km1)
                        # Bound the increment   TODO Replace with a relaxation ???
                        dlamb = min(max(dlamb, -dlamb_max), dlamb_max)
                        lamb += dlamb

                # Display information
                self.monitor(err_alpha, lamb, lamb - lamb_0, Gamma_k, dGamma_k)
                # Increment the iteration counter
                k += 1
                # Check for the end conditions
                if crack_growth_too_small and lamb_k > 1:
                    print(
                        " * The crack is not growing while the load factor is above 1."
                    )
                    converged = True
                    end = True
            # Increment time
            t += 1
            # Apply post processing
            self.postprocessor.postprocess()
            # Export the results
            self.exporter.export(t)
        # End export
        self.exporter.end()
