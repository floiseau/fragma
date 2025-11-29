from petsc4py import PETSc
import dolfinx
from dolfinx import la
from dolfinx.fem import FunctionSpace

dtype = PETSc.ScalarType


def build_elasticity_nullspace(V: FunctionSpace):
    match V.mesh.geometry.dim:
        case 2:
            return build_2D_elasticity_nullspace(V)
        case 3:
            return build_3D_elasticity_nullspace(V)
        case _:
            raise ValueError("Can not define null space for linear elasticity.")


def build_3D_elasticity_nullspace(V: FunctionSpace):
    """Build PETSc nullspace for 3D elasticity

    https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_elasticity.html
    """

    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=dtype) for i in range(6)]
    b = [b.array for b in basis]

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    # Set the three translational rigid body modes
    for i in range(3):
        b[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1
    b[5][dofs[1]] = -x2

    la.orthonormalize(basis)

    basis_petsc = [
        PETSc.Vec().createWithArray(x[: bs * length0], bsize=3, comm=V.mesh.comm)  # type: ignore
        for x in b
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)  # type: ignore


def build_2D_elasticity_nullspace(V: FunctionSpace):
    """Build PETSc nullspace for 2D elasticity

    https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_elasticity.html
    """

    # Get the dimension
    dim = V.mesh.geometry.dim
    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=dtype) for i in range(dim + 1)]
    b = [b.array for b in basis]

    # Get dof indices for each subspace (x and y dofs)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(dim)]

    # Set the three translational rigid body modes
    for i in range(dim):
        b[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1 = x[dofs_block, 0], x[dofs_block, 1]
    b[2][dofs[0]] = -x1
    b[2][dofs[1]] = x0

    la.orthonormalize(basis)

    basis_petsc = [
        PETSc.Vec().createWithArray(x[: bs * length0], bsize=dim, comm=V.mesh.comm)  # type: ignore
        for x in b
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)  # type: ignore
