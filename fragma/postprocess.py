from dolfinx import fem
import ufl


class PostProcessor:
    def __init__(self, domain, model, state):
        """Initialize the post-processing."""
        # Initialize the post expressions and functions
        self.exprs = {}
        self.funcs = {}
        # Initialize strain export
        self.__initialize_strain(domain, model, state)
        # Initialize stress export
        self.__initialize_stress(domain, model, state)

    def __initialize_strain(self, domain, model, state):
        # Compute the strain from ufl
        eps_ufl = model.eps(state)
        # Generate FEM space for strain
        eps_elem = ufl.TensorElement(
            "DG", domain.ufl_cell(), 0, shape=eps_ufl.ufl_shape
        )
        V_eps = fem.FunctionSpace(domain, eps_elem)
        # Convert the strain into an expression
        self.exprs["eps"] = fem.Expression(
            eps_ufl, V_eps.element.interpolation_points()
        )
        # Set the strain function
        self.funcs["eps"] = fem.Function(V_eps, name="Strain")
        self.funcs["eps"].interpolate(self.exprs["eps"])

    def __initialize_stress(self, domain, model, state):
        # Compute the stress from ufl
        sig_ufl = model.sig(state)
        # Generate FEM space for stress
        sig_elem = ufl.TensorElement(
            "DG", domain.ufl_cell(), 0, shape=sig_ufl.ufl_shape
        )
        V_sig = fem.FunctionSpace(domain, sig_elem)
        # Convert the stress into an expression
        self.exprs["sig"] = fem.Expression(
            sig_ufl, V_sig.element.interpolation_points()
        )
        # Set the stress function
        self.funcs["sig"] = fem.Function(V_sig, name="Stress")
        self.funcs["sig"].interpolate(self.exprs["sig"])

    def postprocess(self):
        """Compute quantities after solving an iteration.

        This method computes the strain and stress fields in the domain.
        """
        for func, expr in zip(self.funcs.values(), self.exprs.values()):
            func.interpolate(expr)
