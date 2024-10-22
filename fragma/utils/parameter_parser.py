import sympy as sp

from dolfinx import io, fem


def parse_parameter(par, domain):
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

    Returns
    -------
    par_value : int, float, or dolfinx.Function
        The parsed parameter. If the parameter is a number, it will be returned
        as is. If it's a SymPy expression, it will be represented as a finite
        element function.
    """
    # Check if the parameter is a number
    if isinstance(par, (int, float)):
        # Return the parameter as is
        return par
    else:
        # Declare the coordinate symbol
        x = sp.Symbol("x")
        # Parse the expression using sympy
        par_lambda = sp.utilities.lambdify(x, par, "numpy")
        # Define the function space
        V_par = fem.functionspace(domain.mesh, ("DG", 0))
        # Create the fem function
        par_func = fem.Function(V_par)
        par_func.interpolate(par_lambda)
        # Export the function
        vtk_file = io.VTKFile(
            domain.mesh.comm, "results/heterogeneous_parameter.pvd", "w"
        )
        vtk_file.write_function(par_func, 0)
        vtk_file.close()
        # Return the fem function
        return par_func
