import numpy as np
from petsc4py import PETSc

from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

from utils.build_nullspace import build_elasticity_nullspace
from utils.snes_problem import SNESProblem


class DisplacementSubProblem:
    def __init__(self, pars, domain, state, model):
        # Store the displacement increments
        self.u_incs = pars["loading"]["u_incs"]
        # Initialize the boundary conditions
        bcs_u = self.initialize_boundary_conditions(pars, domain, state)
        # Define the linear problem
        self.define_problem(domain, state, model, bcs_u)

    def initialize_boundary_conditions(self, pars, domain, state):
        bcs_u = []
        # Define a lock point
        bcs_u += self.define_lock_point(pars, domain, state)
        # Define the boundary conditions functions
        bcs_u += self.define_boundary_condition_functions(domain, state)
        # Return the boundary conditions
        return bcs_u

    def define_lock_point(self, pars, domain, state):
        # Get the position of the point
        x0 = pars.get("loading", {}).get("lock_point", None)
        # Return if there is not lock point
        if x0 is None:
            return []
        # Get the state variable
        u = state["u"]
        # Get the function space of the state variable
        V_u = u.function_space

        # Generate the location function
        def lock_point(x):
            return (
                np.isclose(x[0], x0[0])
                & np.isclose(x[1], x0[1])
                & np.isclose(x[2], x0[2])
            )

        # Generate the zero displacement vector
        u_zero = np.array((0,) * domain.mesh.geometry.dim, dtype=default_scalar_type)
        # Locate the dof
        dofs = fem.locate_dofs_geometrical(V_u, lock_point)
        # Generate the Dirichlet boundary condition
        return [fem.dirichletbc(u_zero, dofs, V_u)]

    def define_boundary_condition_functions(self, domain, state):
        print("\n████ DEFINITION OF THE DISPLACEMENT BOUNDARY CONDITIONS")
        # Get the state variable
        u = state["u"]
        # Get the dimensions of domain and facets
        dim = domain.mesh.geometry.dim
        fdim = domain.mesh.geometry.dim - 1
        # Get the displacement function space
        V_u = u.function_space
        # Get boundary facets
        boundary_facets = domain.boundary_facets
        # Get boundary dofs (per comp)
        boundary_dofs = {
            f"{facet_name}_{comp}": fem.locate_dofs_topological(
                (V_u.sub(comp), V_u.sub(comp).collapse()[0]),
                fdim,
                boundary_facet,
            )
            for comp in range(dim)
            for facet_name, boundary_facet in boundary_facets.items()
        }

        print("\n████ INITIALIZE DISPLACEMENT BOUNDARY CONDITIONS")
        # Create variables to store bcs and loading functions
        bcs_u = []
        self.load_funcs = {}
        # Iterage through the displacement increments
        for facet_name, u_inc in self.u_incs.items():
            # Get the component number
            comp = int(facet_name.split("_")[-1])
            # Define an FEM function (to control the BC)
            self.load_funcs[facet_name] = fem.Function(V_u.sub(comp).collapse()[0])
            # Update the load
            with self.load_funcs[facet_name].vector.localForm() as bc_local:
                bc_local.set(u_inc)
            # Add the boundary conditions to the list
            bcs_u.append(
                fem.dirichletbc(
                    self.load_funcs[facet_name], boundary_dofs[facet_name], V_u
                )
            )
        return bcs_u

    def update_boundary_conditions(self, t: float):
        print("Update displacement boundary conditions")
        # Iterate through the load functions
        for facet_name, load_func in self.load_funcs.items():
            # Increment the load function
            with load_func.vector.localForm() as bc_local:
                bc_local.set(default_scalar_type(t * self.u_incs[facet_name]))

    def define_problem(self, domain, state, model, bcs_u):
        print("\n████ DEFINITION OF THE DISPLACEMENT PROBLEM")
        # Get the state variables
        u = state["u"]
        # Get the function spaces
        V_u = u.function_space
        # Define the energy
        energy = model.energy(state, domain.mesh)
        # Derivative of the energy with respect to displacement to obtain the linear problem to determine the stationary point
        E_u = ufl.derivative(energy, u, ufl.TestFunction(V_u))
        E_du = ufl.replace(E_u, {u: ufl.TrialFunction(V_u)})
        # Define the displacement problem
        problem_u = LinearProblem(
            a=ufl.lhs(E_du),
            L=ufl.rhs(E_du),
            bcs=bcs_u,
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
        ns = build_elasticity_nullspace(V_u)
        problem_u.A.setNearNullSpace(ns)
        problem_u.A.setOption(PETSc.Mat.Option.SPD, True)  # type: ignore
        # Display information about the displacement solver
        problem_u.solver.view()
        # Store the problem
        self.problem_u = problem_u

    def update(self, t: float):
        # Update boundary conditions
        self.update_boundary_conditions(t)

    def solve(self):
        self.problem_u.solve()


class CrackPhaseSubProblem:
    def __init__(self, pars, domain, state, model):
        # Define the boundary conditions functions
        bcs_alpha = self.define_boundary_condition_functions(domain, state)
        # Define the crack phase problem
        self.define_problem(domain, state, model, bcs_alpha)

    def define_problem(self, domain, state, model, bcs_alpha):
        print("\n████ DEFINITION OF THE CRACK PHASE PROBLEM")
        # Get the state variables
        alpha = state["alpha"]
        # Store the state variable
        self.alpha = alpha
        # Get the function spaces
        V_alpha = alpha.function_space
        # Define the energy
        energy = model.energy(state, domain.mesh)
        # Derivative of the energy with respect to crack phase
        E_alpha = ufl.derivative(energy, alpha, ufl.TestFunction(V_alpha))
        E_alpha_alpha = ufl.derivative(E_alpha, alpha, ufl.TrialFunction(V_alpha))

        # Define the crack phase problem
        snes_problem_alpha = SNESProblem(E_alpha, E_alpha_alpha, alpha, bcs_alpha)
        # Initialize the LHS
        b = fem.petsc.create_vector(snes_problem_alpha.L)
        # Initialize the jacobian
        J = fem.petsc.create_matrix(fem.form(snes_problem_alpha.a))

        # Create Newton solver
        problem_alpha = PETSc.SNES().create()
        problem_alpha.setType("vinewtonrsls")
        problem_alpha.setFunction(snes_problem_alpha.F, b)
        problem_alpha.setJacobian(snes_problem_alpha.J, J)
        problem_alpha.setTolerances(rtol=1.0e-9, max_it=50)

        problem_alpha.getKSP().setType("preonly")
        problem_alpha.getKSP().setTolerances(rtol=1.0e-9)
        problem_alpha.getKSP().getPC().setType("lu")
        problem_alpha.getKSP().getPC().setFactorSolverType("mumps")

        # Define lower and upper bounds functions for the crack phase field
        self.alpha_lb = fem.Function(V_alpha, name="Lower bound")
        self.alpha_ub = fem.Function(V_alpha, name="Upper bound")
        # Set the lower bound
        with self.alpha_lb.vector.localForm() as alpha_lb_local:
            alpha_lb_local.set(0.0)
        fem.set_bc(self.alpha_lb.vector, bcs_alpha)
        # Set the upper bound
        one_alpha = fem.Function(V_alpha)
        with self.alpha_ub.vector.localForm() as alpha_ub_local:
            alpha_ub_local.set(1.0)
        fem.set_bc(self.alpha_ub.vector, bcs_alpha)
        # Set the crack phrase boundary bound (Note: they are passed as reference and not as values)
        problem_alpha.setVariableBounds(self.alpha_lb.vector, self.alpha_ub.vector)

        # Initialise alpha
        self.alpha_lb.vector.copy(alpha.vector)

        # Display information about the displacement solver
        problem_alpha.view()
        # Store the problem on alpha in subproblems
        self.problem_alpha = problem_alpha

    def define_boundary_condition_functions(self, domain, state):
        print("\n████ INITIALISATION OF THE CRACK FIELD")
        # Add the damage boundary conditions if there is an initial crack
        if "crack" in domain.boundary_facets:
            # Get the dimensions of domain and facets
            dim = domain.mesh.geometry.dim
            fdim = domain.mesh.geometry.dim - 1
            # Get the crack phase function space
            V_alpha = state["alpha"].function_space
            # Get boundary facets
            boundary_facets = domain.boundary_facets
            # Get boundary dofs (per comp)
            boundaries = {
                f"{facet_name}": fem.locate_dofs_topological(
                    V_alpha,
                    fdim,
                    boundary_facet,
                )
                for facet_name, boundary_facet in boundary_facets.items()
            }
            # Create the crack boundary condition
            bcs_alpha_crack = [fem.dirichletbc(1.0, boundaries["crack"], V_alpha)]
            # Create the uncrackable crack boundary condition
            bcs_alpha_noncrackable = [
                fem.dirichletbc(0.0, b_dof, V_alpha)
                for boundary, b_dof in boundaries.items()
                if boundary.startswith("non-crackable")
            ]
            #
            bcs_alpha = bcs_alpha_crack + bcs_alpha_noncrackable
        else:
            bcs_alpha = []
        return bcs_alpha

    def update_boundary_conditions(self, t: float):
        ...

    def update(self, t: float):
        # Update the crack phase lower bound
        self.alpha.vector.copy(self.alpha_lb.vector)
        # Update of boundary conditions ?
        self.update_boundary_conditions(t)

    def solve(self):
        self.problem_alpha.solve(None, self.alpha.vector)


class CrackPhaseSubProblemMiehe(CrackPhaseSubProblem):

    def define_problem(self, domain, state, model, bcs_alpha):
        print("\n████ DEFINITION OF THE CRACK PHASE PROBLEM")
        # Get the state variables
        alpha = state["alpha"]
        # Store the state variable
        self.alpha = alpha
        # Get the function spaces
        V_alpha = alpha.function_space
        # Define the energy
        energy = model.energy(state, domain.mesh)
        # Derivative of the energy with respect to crack phase
        E_alpha = ufl.derivative(energy, alpha, ufl.TestFunction(V_alpha))
        E_dalpha = ufl.replace(E_alpha, {alpha: ufl.TrialFunction(V_alpha)})
        # Define the displacement problem
        problem_alpha = LinearProblem(
            a=ufl.lhs(E_dalpha),
            L=ufl.rhs(E_dalpha),
            bcs=bcs_alpha,
            u=alpha,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_solver_type": "mumps",
            },
        )
        # Display information about the displacement solver
        problem_alpha.solver.view()
        # Store the problem
        self.problem_alpha = problem_alpha

    def update(self, t: float):
        # Update of boundary conditions ?
        self.update_boundary_conditions(t)

    def solve(self):
        self.problem_alpha.solve()
