import sympy as sp

from dolfinx import io, fem


def parse_parameter(par, domain, export_lambda: bool = False):
    """
    Parse the given parameter.

    If the parameter is a number (integer or float), returns the raw number.
    Otherwise, it interprets the parameter as a mathematical expression,
    parses it using SymPy, and creates a finite element function representing
    the parsed expression on the given domain.

    Parameters
    ----------
    par : int, float, or sympy.Expr
        The parameter to parse. If it's a number, it will be returned as is.
        If it's a SymPy expression, it will be parsed and represented as a
        finite element function.
    domain : fragma.Domain.domain
        The domain on which to interpolate the parsed parameter.
    export_lambda: bool
        Export the lambda function used to interpolate the FEM function.
        It is a function of both space and time. This is particularly
        useful for time varying parameters (thermal load for instance).

    Returns
    -------
    par_value : int, float, or dolfinx.Function
        The parsed parameter. If the parameter is a number, it will be returned
        as is. If it's a SymPy expression, it will be represented as a finite
        element function.
    par_lambda : Optional, function
        The python function used to interpolate the FEM function.
        This is a function of both space and time.
    """
    # Check if the parameter is a number
    if isinstance(par, (int, float)):
        # Return the parameter as is
        return par
    else:
        # Declare the coordinate symbol
        x, t = sp.Symbol("x"), sp.Symbol("t")
        # Parse the expression using sympy
        par_lambda = sp.utilities.lambdify([x, t], par, "numpy")
        # Define the function space
        V_par = fem.functionspace(domain.mesh, ("DG", 0))
        # Create the fem function
        par_fem_func = fem.Function(V_par)
        par_fem_func.interpolate(lambda xx: par_lambda(xx, 0))
        # Export the function
        vtk_file = io.VTKFile(
            domain.mesh.comm, "results/heterogeneous_parameter.pvd", "w"
        )
        vtk_file.write_function(par_fem_func, 0)
        vtk_file.close()
        # Return the fem function
        if not export_lambda:
            return par_fem_func
        else:
            return par_fem_func, par_lambda


def parse_boundary_condition(bc_expr):
    # Declare the coordinate symbol
    x, l = sp.Symbol("x"), sp.Symbol("l")
    # Parse the expression using sympy
    bc_expr_sp = sp.utilities.lambdify([x, l], bc_expr, "numpy")
    # Return the fem function
    return bc_expr_sp
