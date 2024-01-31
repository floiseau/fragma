import time
import re

from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
from scipy.optimize import root_scalar

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
            print(f" | Crack length increment       : {dGamma:.3g}")
            print(f" | Final crack length           : {Gamma:.3g}")
            print(f" | New load factor              : {lamb:.3g}")
            print(f" | New load fractor inc         : {dlamb:.3g}")

    def compute_initial_load_factor(self, lamb_func):
        print(f"--- Initialize the load factor")
        print(f"* Perform an elastic prediction")
        # Update boundary conditions with load factor
        self.subproblems["u"].update(1)
        # Solve the displacement problem
        self.subproblems["u"].solve()
        print(f"* Compute the initial load factor")
        # Get the crack phase varaible
        alpha = self.state["alpha"]
        # Get the parameters
        Gc = self.model.Gc
        ell = self.model.ell
        cw = self.model.cw()
        wp = self.model.wp(alpha)
        ap = self.model.ap(alpha)
        # Compute stress
        sig = self.model.sig(self.state)
        # Compute strain
        eps = self.model.eps(self.state)
        # Update lambda_expression
        lamb_expr = fem.Expression(
                ufl.sqrt(-2*Gc/(cw*ell) * wp / (ap*ufl.inner(sig, eps))),
                lamb_func.function_space.element.interpolation_points())
        # Define the history field
        lamb_func.interpolate(lamb_expr)
        # Compute the initial load factor
        lamb_array = lamb_func.vector[:]
        # # Filter out the region where alpha is already high
        # lamb_array[alpha.vector[:]>0.5] = 1e12
        # Compute the initial load factor
        lamb = np.min(lamb_array)
        # ### Remove the region that are not breakable
        # # Get the domain
        # domain = self.domain
        # # Get the dimensions of domain and facets
        # dim = domain.mesh.geometry.dim
        # fdim = domain.mesh.geometry.dim - 1
        # # Get boundary facets
        # boundary_facets = domain.boundary_facets
        # # Get boundary dofs (per comp)
        # boundaries = {
        #     f"{facet_name}": fem.locate_dofs_topological(
        #         lamb_func.function_space,
        #         fdim,
        #         boundary_facet,
        #     )
        #     for facet_name, boundary_facet in boundary_facets.items()
        # }
        # for name, mask in boundaries.items():
        #     if name.startswith("non-crackable"):
        #         lamb_array[mask] = 1e12
        print(f" | Initial load factor: {lamb:0.4g}")
        return lamb

    def solve_displacement_crack_phase_problem(self, lamb):
        # Update boundary conditions with load factor
        self.subproblems["u"].update(lamb)
        # Get the crack phase state variable
        alpha = self.state["alpha"]
        # Define alpha at previous AM iteration (for error computation)
        alpha_k = fem.Function(alpha.function_space)
        alpha.vector.copy(alpha_k.vector)
        # Initialize the AM iteration counter
        am_iter = 0
        # Initialize the AM convergence flag
        am_converged = False
        # Alternate minimization loop
        print("* Alternate minimization")
        while not am_converged and am_iter < 500:  # Alternate minimization loop
            # Increment the counter
            am_iter += 1
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
                f" | Iteration: {am_iter:03d}; Error: {err_alpha:.4e}",
                end="\r" if not am_converged else "\n",
            )
        if not am_converged:
            raise RuntimeError("The alternate minimization did not converge.")
            

    def compute_constraint_function(self, lamb):
        # Increment the evaluation counter
        self.k += 1
        # Display informations
        print(f"\n--- Evaluation {self.k} with lambda={lamb:.3g}")
        # Solve the displacement-crack phase problem
        self.solve_displacement_crack_phase_problem(lamb)
        # Get the crack phase variable
        alpha = self.state["alpha"]
        # Compute the crack surface
        dx = ufl.Measure("dx", domain=self.domain.mesh)
        self.Gamma = fem.assemble_scalar(fem.form(alpha * dx))
        # Compute the crack surface increment
        dGamma_k = self.Gamma - self.Gamma_0
        # Compute the constraint
        C = dGamma_k - self.dGamma
        if MPI.COMM_WORLD.rank == 0:
            print(f"* Summary of evaluation:")
            print(f" | Constraint value       : {C:.3g}")
            print(f" | Crack length increment : {dGamma_k:.3g}")
            print(f" | Crack length           : {self.Gamma:.3g}")
        return C

    def solve(self):
        print("\n████ RESOLUTION")
        # Initialize calculation of load factor
        # element_lamb = ufl.FiniteElement("DG", self.domain.mesh.ufl_cell(), degree=0)
        element_lamb = ufl.FiniteElement("P", self.domain.mesh.ufl_cell(), degree=1)
        V_lamb = fem.FunctionSpace(self.domain.mesh, element_lamb)
        lamb_func = fem.Function(V_lamb, name="Load factor")
        # Get the state
        u, alpha = self.state["u"], self.state["alpha"]
        # Set the crack surface increment
        self.dGamma = 5e-7
        # Set the tolerance on the crack surface increment
        tol_dGamma = self.dGamma/2
        # Set the initial interval size for the load factor
        # NOTE: the choice of dlamb is directly linked to tol_dGamma
        # NOTE: dlamb too small may be better than too high
        dlamb = 0.01
        # Initialize the load factor
        lamb = 0
        # Initialize time step
        t = 0
        # Initialize alpha (diffuse pre-crack)
        print("* Initialize the solution")
        self.solve_displacement_crack_phase_problem(0)
        # Initialize the crack surface
        dx = ufl.Measure("dx", domain=self.domain.mesh)
        self.Gamma = fem.assemble_scalar(fem.form(alpha * dx))
        # Initialize time load loop
        end = False
        while not end:  # Time steps loop
            # Increment time
            t += 1
            # Display information
            print(f"\n\n== Time step {t:08d}")
            # # Compute the initial load factor
            # lamb = self.compute_initial_load_factor(lamb_func)
            # Update the initial Gamma
            self.Gamma_0 = self.Gamma
            # Update the crack phase lower bound
            self.subproblems["alpha"].update(t)
            # Initialize an evaluation counter
            self.k = 0
            # Set the convergence flag
            converged = False
            while not converged:
                # Set the bounds
                lamb_min = lamb
                lamb_max = lamb + dlamb
                # Solve the non-linear load factor problem
                try:
                    res = root_scalar(
                            self.compute_constraint_function,
                            bracket=(lamb_min, lamb_max),
                            xtol = tol_dGamma,
                            method="toms748",
                            # disp=True,
                            )
                    # Store the result
                    lamb = res.root
                    # Set the convergence flag to True
                    converged = True
                except ValueError as e:
                    # Get the error message
                    error_message = str(e)
                    # If the error is not a bracketing error
                    if not "f(a) and f(b) must have different signs" in error_message:
                        raise e
                    # Display some information
                    print("* Load factor not in the interval.")
                    if lamb_max > 1:
                        print(" | The next adequate load factor is above 1.")
                        print(" | Stopping the resolution.")
                        converged = True
                        end = True
                    else:
                        # Get the signs
                        C_a, C_b = self.extract_signs_from_error(error_message)
                        # Check the sign of C_a and C_b
                        if C_a > 0 and C_b > 0:   # C_a and C_b are positive
                            print(" | Decreasing the load factor.")
                            # Translate "down" the load factor interval
                            lamb -= dlamb
                        elif C_a < 0 and C_b < 0: # C_a and C_b are negative
                            # Translate "up" the load factor interval
                            print(" | Increasing the load factor.")
                            lamb += dlamb
                        else:
                            raise ValueError("Error in the adapation of the interval.")
            # Apply post processing
            self.postprocessor.postprocess()
            # Export the results
            self.exporter.export(t)
        # End export
        self.exporter.end()

    def extract_signs_from_error(self, error_message):
        # Check if the error message indicates that both f(a) and f(b) have the same sign
        sign_match = re.search(r'but f\((.*?)\)=(.*?), f\((.*?)\)=(.*?)$', error_message)
        if sign_match:
            a, fa, b, fb = sign_match.groups()
            return float(fa), float(fb)
        else:
            # If parsing fails, return None or raise an exception, depending on your preference
            return None

    # def solve(self):
    #     print("\n████ RESOLUTION")
    #     # Get integrand
    #     dx = ufl.Measure("dx", domain=self.domain.mesh)
    #     # Get the state
    #     u, alpha = self.state["u"], self.state["alpha"]
    #     # Initialize alpha (diffuse pre-crack)
    #     self.subproblems["alpha"].solve()
    #     # Set the crack surface increment
    #     dGamma = 5e-7
    #     # Set the tolerance on the crack surface increment
    #     tol_dGamma = 2.5e-7
    #     # Set load factor changes bounds
    #     dlamb_max = 0.02
    #     # Initialize time step
    #     t = 0
    #     # Initialize the load factor
    #     lamb_k = 0.001
    #     # Initialize the crack surface
    #     Gamma_k = fem.assemble_scalar(fem.form(alpha * dx))
    #     # Define alpha at previous iteration (k) (for error computation)
    #     alpha_k = fem.Function(alpha.function_space)
    #     alpha.vector.copy(alpha_k.vector)
    #     # Initialize time load loop
    #     end = False
    #     while not end:  # Time steps loop
    #         # Increment time
    #         t += 1
    #         # Display information
    #         print(f"\n== Time step {t:08d}")
    #         # Update the crack phase lower bound
    #         self.subproblems["alpha"].update(t)
    #         # Store the crack surface at begin of load step
    #         Gamma_0 = Gamma_k
    #         # Store the load factor at the begin of load step
    #         lamb_0 = lamb_k
    #         # Initialize the new load factor to the one at the begin of the step
    #         lamb = lamb_0
    #         # Initialize the constraint
    #         C_k = None
    #         # Initialize the iteration
    #         converged = False
    #         k = 0
    #         while not converged:  # Iteration loop
    #             # Increment the iteration counter
    #             k += 1
    #             # Store the load factor from previous iteration (km1)
    #             lamb_km1 = lamb_k
    #             # Set the load factor for the current iteration
    #             lamb_k = lamb
    #             # Display information
    #             print(
    #                 f"--- Iteration {k} with lambda={lamb_k:.3g} (dlambda={lamb_k-lamb_0:.3})"
    #             )

    #             # Update boundary conditions with load factor
    #             self.subproblems["u"].update(lamb_k)

    #             # Solve the u and alpha problems
    #             am_converged = False
    #             k_sub = 0
    #             print("* Alternate minimization")
    #             while not am_converged and k_sub < 500:  # Alternate minimization loop
    #                 # Increment the counter
    #                 k_sub += 1
    #                 # Solve the displacement problem
    #                 self.subproblems["u"].solve()
    #                 # Store previous alpha
    #                 alpha.vector.copy(alpha_k.vector)
    #                 # Solve the crack phase problem
    #                 self.subproblems["alpha"].solve()
    #                 # Check error (L2)
    #                 err_alpha = np.max(alpha.vector[:] - alpha_k.vector[:])
    #                 # Update alpha_k
    #                 alpha.vector.copy(alpha_k.vector)
    #                 # Check convergence
    #                 am_converged = err_alpha <= 1e-5
    #                 # Print some information
    #                 print(
    #                     f" | Iteration: {k_sub:03d}; Error: {err_alpha:.4e}",
    #                     end="\r" if not am_converged else "\n",
    #                 )

    #             if not am_converged:
    #                 print(" | Did not converge, retry with a smaller increment.")
    #                 lamb -= abs(lamb_k - lamb_0) / 2
    #                 continue

    #             # Compute the crack surface
    #             Gamma_km1 = Gamma_k
    #             Gamma_k = fem.assemble_scalar(fem.form(alpha * dx))
    #             dGamma_k = Gamma_k - Gamma_0
    #             # Compute the constraint
    #             C_km1 = C_k
    #             C_k = Gamma_k - (Gamma_0 + dGamma)

    #             # Check crack growth
    #             crack_growth_too_small = dGamma_k < dGamma - tol_dGamma
    #             crack_growth_too_large = dGamma_k > dGamma + tol_dGamma
    #             crack_growth_ok = (
    #                 not crack_growth_too_small and not crack_growth_too_large
    #             )

    #             print("* Load control:")
    #             if crack_growth_ok:  # Keep same load factor
    #                 print(" | Crack growth is OK.")
    #                 print(" | Keeping the same load factor for next load step.")
    #                 # The crack growth is small enought
    #                 converged = True
    #             else:  # Load factor is too large
    #                 # Crack growth is too large
    #                 converged = False
    #                 if crack_growth_too_small:
    #                     print(" | Crack growth is too small.")
    #                 else:
    #                     print(" | Crack growth is too high.")

    #                 if k == 1:  # First iteration
    #                     print(" | First iteration, perturbate the load factor")
    #                     # Perturbate the load factor
    #                     lamb += 0.01 * abs(lamb)
    #                 elif C_k - C_km1 == 0:  # Crack length did not change
    #                     print(" | No crack growth, increase the load factor")
    #                     # Increase the load factor (double the current iteration increment)
    #                     # NOTE This is complicated to manage
    #                     lamb += 1.1 * abs(lamb)
    #                 else:
    #                     print(" | Updating dlamb with secant method")
    #                     # Compute the new increment of load factor
    #                     dlamb = -C_k * (lamb_k - lamb_km1) / (C_k - C_km1)
    #                     # Bound the increment   TODO Replace with a relaxation ???
    #                     dlamb = min(max(dlamb, -dlamb_max), dlamb_max)
    #                     # Increment the load factor
    #                     lamb += dlamb

    #             # Display information
    #             self.monitor(err_alpha, lamb, lamb - lamb_0, Gamma_k, dGamma_k)
    #             # Check for the end conditions
    #             if crack_growth_too_small and lamb_k > 1:
    #                 print(
    #                     " * The crack is not growing while the load factor is above 1."
    #                 )
    #                 converged = True
    #                 end = True
    #         # Apply post processing
    #         self.postprocessor.postprocess()
    #         # Export the results
    #         self.exporter.export(t)
    #     # End export
    #     self.exporter.end()
