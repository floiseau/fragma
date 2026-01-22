from typing import List

from petsc4py import PETSc
from mpi4py import MPI

import dolfinx
import ufl


class SNESProblem:
    """Nonlinear problem class compatible with PETSC.SNES solver.

    Ressources:
        https://fenicsproject.discourse.group/t/set-bounds-in-a-nonlinearproblem/7993/2
        https://github.com/FEniCS/dolfinx/blob/f55eadde9bba6272d5a111aac97bcb4d7f2b5231/python/test/unit/nls/test_newton.py#L156
    """

    def __init__(
        self,
        F: ufl.form.Form,
        J: ufl.form.Form,
        u: dolfinx.fem.Function,
        bcs: List[dolfinx.fem.DirichletBC],
    ):
        """This class set up structures for solving a non-linear problem using Newton's method.

        Parameters
        ==========
        F: Residual.
        J: Jacobian.
        u: Solution.
        bcs: Dirichlet boundary conditions.
        """
        self.L = dolfinx.fem.form(F)
        self.a = dolfinx.fem.form(J)
        self.bcs = bcs
        # self._F, self._J = None, None
        self.u = u

        # Create matrix and vector to be used for assembly of the non-linear problem
        self.b = dolfinx.fem.petsc.create_vector(
            dolfinx.fem.extract_function_spaces(self.L)
        )
        self.A = dolfinx.fem.petsc.create_matrix(self.a)

    def F(self, snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble the residual F into the vector b.

        Parameters
        ==========
        snes: the snes object
        x: Vector containing the latest solution.
        b: Vector to assemble the residual into.
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # Reset the residual
        with b.localForm() as f_local:
            f_local.set(0.0)
        # Assemble the vector
        dolfinx.fem.petsc.assemble_vector(b, self.L)
        # Apply boundary conditions
        dolfinx.fem.petsc.apply_lifting(b, [self.a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, self.bcs, x, -1.0)

    def J(self, snes, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat):
        """Assemble the Jacobian matrix.

        Parameters
        ==========
        x: Vector containing the latest solution.
        A: Matrix to assemble the Jacobian into.
        """
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, self.a, self.bcs)
        A.assemble()


class TAOProblem:
    """Nonlinear optimization problem class compatible with PETSc.TAO solver.

    For more info, check: https://fenicsproject.discourse.group/t/parallel-computation-tao-solver/12768/5.
    """

    def __init__(self, f, g, H, u, bcs):
        """This class sets up structures for solving an optimization problem using TAO.

        Parameters
        ==========
        f: The scalar objective function.
        g: Gradient of the objective function.
        H: Hessian of the objective function.
        u: Solution (state variable).
        bcs: Dirichlet boundary conditions.
        """
        # Storage of the arguments
        self.bcs = bcs
        self.u = u
        self.obj = dolfinx.fem.form(f)
        self.L = dolfinx.fem.form(g)
        self.a = dolfinx.fem.form(H)

        # Create matrix and vector for gradient and Hessian assembly
        self.b = dolfinx.fem.petsc.create_vector(
            dolfinx.fem.extract_function_spaces(self.L)
        )
        self.A = dolfinx.fem.petsc.create_matrix(self.a)

    def f(self, tao, x):
        """Evaluate the objective function at the current solution.

        Parameters
        ==========
        tao: TAO object (ignored here, but required by interface).
        x: Current solution vector.

        Returns
        =======
        Objective value (float).
        """
        # # Connection between ranks in the vector x: FORWARD
        # x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # # We copy the vector x in the vector u
        # x.copy(self.u.x.petsc_vec)
        # # Connection between ranks in the vector u in the class
        # self.u.x.petsc_vec.ghostUpdate(
        #     addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        # )
        local_res = dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.obj))
        global_res = self.u.function_space.mesh.comm.allreduce(local_res, op=MPI.SUM)

        return global_res

    def F(self, tao, x, F):
        """Assemble the gradient (residual) into the vector g.

        Parameters
        ==========
        tao: TAO object (ignored here, but required by interface).
        x: Current solution vector.
        F: Vector to assemble the gradient into.
        """
        #  # Connection between ranks in the vector x: FORWARD
        #  x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        #  # We copy the vector x in the vector u
        #  x.copy(self.u.x.petsc_vec)
        #  # Connection between ranks in the vector u in the class
        #  self.u.x.petsc_vec.ghostUpdate(
        #      addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        #  )

        with F.localForm() as f_local:
            f_local.set(0.0)

        dolfinx.fem.petsc.assemble_vector(F, self.L)
        # Include the boundary conditions using the lifting technique
        dolfinx.fem.petsc.apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        # Connection between ranks in the vector x: REVERSE
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        # Redefine the function F with the boundary condition.
        dolfinx.fem.petsc.set_bc(F, self.bcs, x, -1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def J(self, tao, x, A, P):
        """Assemble the Hessian matrix.

        Parameters
        ==========
        tao: TAO object (ignored here, but required by interface).
        x: Current solution vector.
        A: Matrix to assemble the Hessian into.
        P: (required by the interface).
        """
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, self.a, self.bcs)
        A.assemble()
