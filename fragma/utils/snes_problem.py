from typing import List

from petsc4py import PETSc

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
        self._F, self._J = None, None
        self.u = u

        # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self.A = dolfinx.fem.create_matrix(self.a)
        self.b = dolfinx.fem.create_vector(self.L)

    def F(self, snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble the residual F into the vector b.

        Parameters
        ==========
        snes: the snes object
        x: Vector containing the latest solution.
        b: Vector to assemble the residual into.
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # Reset the residual
        with b.localForm() as f_local:
            f_local.set(0.0)
        # Assemble the vector
        dolfinx.fem.petsc.assemble_vector(b, self.L)
        # Apply boundary conditions
        dolfinx.fem.apply_lifting(b, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, self.bcs, x, -1.0)

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
