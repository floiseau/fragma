import numpy as np
import sympy as sp


from dolfinx import fem, default_scalar_type
import ufl


class BaseModel:
    """
    Base class for defining material models.

    This class provides common functionalities and utilities for material models.

    Parameters
    ----------
    pars : dict
        Dictionary containing parameters of the material model.

    Attributes
    ----------
    la : dolfinx.Constant
        Lame coefficient lambda.
    mu : dolfinx.Constant
        Lame coefficient mu.
    """

    def __init__(self, pars, domain):
        """
        Initialize the BaseModel.

        Parameters
        ----------
        pars : dict
            Dictionary containing parameters of the material model.
        domain : fragma.Domain.domain
            Domain object used to initialize heterogeneous properties.
        """
        # Get elastic parameters
        self.E = self.parse_parameter(pars["mechanical"]["E"], domain)
        self.nu = self.parse_parameter(pars["mechanical"]["nu"], domain)
        # Compute Lame coefficient
        self.la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))
        # Check the 2D assumption
        if pars["model"]["dim"] == 2:
            self.assumption = pars["model"]["2D_assumption"]
            match self.assumption:
                case "plane_stress":
                    print("Plane stress assumption")
                    self.la = 2 * self.mu * self.la / (self.la + 2 * self.mu)
                case "plane_strain":
                    print("Plane strain assumption")
                case _:
                    raise ValueError(
                        f'The 2D assumption "{self.assumption}" in unknown'
                    )

    def parse_parameter(self, par, domain):
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
            par_elem = ufl.FiniteElement("DG", domain.mesh.ufl_cell(), 0)
            V_par = fem.FunctionSpace(domain.mesh, par_elem)
            # Create the fem function
            par_func = fem.Function(V_par)
            par_func.interpolate(par_lambda)
            # Return the fem function
            return par_func

    def eps(self, state):
        """
        Compute the strain tensor.

        Parameters
        ----------
        state : dict
            Dictionary containing state variables.

        Returns
        -------
        ufl.form.Expression
            Strain tensor.
        """
        return ufl.sym(ufl.grad(state["u"]))

    def sig(self, state):
        """
        Compute the stress tensor.

        Parameters
        ----------
        state : dict
            Dictionary containing state variables.

        Returns
        -------
        ufl.form.Expression
            Stress tensor.
        """
        # Get elastic parameters
        mu, la = self.mu, self.la
        # Get the state variables
        u = state["u"]
        # Compute the stess
        return la * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0 * mu * self.eps(state)

    def sig_eff(self, state):
        """
        Compute the effective stress tensor.

        Parameters
        ----------
        state : dict
            Dictionary containing state variables.

        Returns
        -------
        ufl.form.Expression
            Effective stress tensor.
        """
        return self.sig(state)

    def energy(self, state, mesh):
        """
        Compute the energy.

        This method should be implemented in the child class.

        Parameters
        ----------
        state : dict
            Dictionary containing state variables.
        mesh : dolfinx.Mesh
            The mesh representing the domain.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the child class.
        """
        raise NotImplementedError(
            "Model: The method 'energy' must be implemented in the child class."
        )


class ElasticModel(BaseModel):
    """
    Material model for linear elasticity.

    This class implements the material model for linear elasticity.

    Parameters
    ----------
    pars : dict
        Dictionary containing parameters of the material model.
    """

    def elastic_energy(self, state, domain):
        """
        Compute the elastic energy.

        Parameters
        ----------
        state : dict
            Dictionary containing state variables.
        domain : Domain
            The domain object representing the computational domain.

        Returns
        -------
        ufl.form.Expression
            Elastic energy.
        """
        # Get the integrands
        dx = ufl.Measure("dx", domain=domain.mesh)
        # Define the total energy
        return 1 / 2 * ufl.inner(self.sig_eff(state), self.eps(state)) * dx

    def energy(self, state, domain):
        """
        Compute the total energy.

        Parameters
        ----------
        state : dict
            Dictionary containing state variables.
        domain : Domain
            The domain object representing the computational domain.

        Returns
        -------
        ufl.form.Expression
            Total energy.
        """
        # Define the energy terms
        elastic_energy = self.elastic_energy(state, domain)
        # Define the total energy
        return elastic_energy


class FractureModel(ElasticModel):
    """
    Material model for fracture mechanics.

    This class implements the material model for fracture mechanics.

    Parameters
    ----------
    pars : dict
        Dictionary containing parameters of the material model.
    """

    def __init__(self, pars, domain):
        """
        Initialize the FractureModel.

        Parameters
        ----------
        pars : dict
            Dictionary containing parameters of the material model.
        """
        # Initialise parent class
        super().__init__(pars, domain)
        # Get the degradation model
        model_par = pars["model"]["model"]
        if model_par in ["AT1", "AT2"]:
            self.deg_model = "AT"
        else:
            self.deg_model = model_par.split("-")[0]
        # Get the dissipation model
        if model_par in ["AT1", "AT2"]:
            self.dis_model = model_par
        else:
            self.dis_model = model_par.split("-")[1]
        # Get the residual crack phase
        self.alpha_res = pars["numerical"]["alpha_res"]
        # Get fracture parameters
        self.ell = self.parse_parameter(pars["mechanical"]["ell"], domain)
        # Check for anisotropy
        self.is_anisotropic = "theta_0" in pars["mechanical"]
        if not self.is_anisotropic:
            # Get the critical energy release rate
            self.Gc = self.parse_parameter(pars["mechanical"]["Gc"], domain)
        else:
            # Get the critical energy release rate (min and max)
            Gc_min = self.parse_parameter(pars["mechanical"]["Gc_min"], domain)
            Gc_max = self.parse_parameter(pars["mechanical"]["Gc_max"], domain)
            # Convert to other model parameters
            self.Gc = np.sqrt(1 / 2 * (Gc_min**2 + Gc_max**2))
            self.aG = 1 / 2 * (Gc_max**2 - Gc_min**2) / self.Gc**2
            # Ge the anisotropy angle
            self.theta_0 = (
                self.parse_parameter(pars["mechanical"]["theta_0"], domain)
                * np.pi
                / 180
                if "theta_0" in pars["mechanical"]
                else 0
            )

    def a(self, alpha):
        """
        Degradation function.

        Parameters
        ----------
        alpha : ufl.form.Expression
            Crack phase.

        Returns
        -------
        ufl.form.Expression
            Degradation function.
        """
        # Residual crack phase
        alpha_res = self.alpha_res
        # Compute a
        match self.deg_model:
            case "AT":
                return (1 - alpha) ** 2 + alpha_res
            case "KKL":
                return 4 * (1 - alpha) ** 3 - 4 * (1 - alpha) ** 3 + alpha_res
            case "KSM":
                return 3 * (1 - alpha) ** 2 - 3 * (1 - alpha) ** 2 + alpha_res
            case _:
                raise ValueError(
                    f"The degradation model named '{self.deg_model}' does not exists."
                )

    def ap(self, alpha):
        """
        Derivative of the degradation function.

        Parameters
        ----------
        alpha : ufl.form.Expression
            Crack phase.

        Returns
        -------
        ufl.form.Expression
            Derivative of the degradation function.
        """
        # Compute w
        match self.deg_model:
            case "AT":
                return -2 * (1 - alpha)
            case "KKL":
                return -12 * (1 - alpha) ** 2 + 12 * (1 - alpha) ** 3
            case "KSM":
                return -6 * (1 - alpha) + 6 * (1 - alpha) ** 2
            case _:
                raise ValueError(
                    f"The degradation model named '{self.deg_model}' does not exists."
                )

    def w(self, alpha):
        """
        Dissipation function.

        Parameters
        ----------
        alpha : ufl.form.Expression
            Crack phase.

        Returns
        -------
        ufl.form.Expression
            Dissipation function.
        """
        # Compute w
        match self.dis_model:
            case "AT1":
                return alpha
            case "AT2":
                return alpha**2
            case "DW":
                return 16 * alpha**2 * (1 - alpha) ** 2
            case _:
                raise ValueError(
                    f"The degradation model named '{self.dis_model}' does not exists."
                )

    def wp(self, alpha):
        """
        Derivative of the dissipation function.

        Parameters
        ----------
        alpha : ufl.form.Expression
            Crack phase.

        Returns
        -------
        ufl.form.Expression
            Derivative of the dissipation function.
        """
        # Compute w
        match self.dis_model:
            case "AT1":
                return 1
            case "AT2":
                return 2 * alpha
            case "DW":
                return 16 * (2 * alpĥa * (1 - alpha) ** 2 - 2 * alpha**2 * (1 - alpha))
            case _:
                raise ValueError(
                    f"The degradation model named '{self.dis_model}' does not exists."
                )

    def cw(self):
        """
        Normalization coefficient.

        Returns
        -------
        float
            Normalization coefficient.
        """
        match self.dis_model:
            case "AT1":
                return 8 / 3
            case "AT2":
                return 2
            case "DW":
                return 4 * 2 / 3
            case _:
                raise ValueError(
                    f"The degradation model named '{self.dis_model}' does not exists."
                )

    def sig_eff(self, state):
        """
        Effective stress accounting the degradation due to the crack phase.

        Parameters
        ----------
        state : dict
            Dictionary containing state variables.

        Returns
        -------
        ufl.form.Expression
            Effective stress.
        """
        return self.a(state["alpha"]) * self.sig(state)

    def fracture_dissipation(self, state, domain):
        """
        Compute the energy dissipated by fracture.

        Parameters
        ----------
        state : dict
            Dictionary containing state variables.
        domain : Domain
            The domain object representing the computational domain.

        Returns
        -------
        ufl.form.Expression
            Energy dissipated by fracture.

        """
        # Get the integrands
        dx = ufl.Measure("dx", domain=domain.mesh)
        # Get state variables
        alpha = state["alpha"]
        # Get the fracture parameters
        Gc = self.Gc
        ell = self.ell
        cw = self.cw()
        # Compute the anisotropy matrix
        A_np = np.eye(domain.mesh.geometry.dim)
        # Add the higher order terms if the model is anisotropic
        if self.is_anisotropic:
            # Get the parameters
            aG, theta_0 = self.aG, self.theta_0
            #  Compute the 2nd order term of the anisotropy tensor
            A_np += aG * np.array(
                [
                    [np.cos(2 * theta_0), np.sin(2 * theta_0)],
                    [np.sin(2 * theta_0), -np.cos(2 * theta_0)],
                ]
            )
        # Create an FEM constant for the anisotropy matrix
        A = fem.Constant(domain.mesh, A_np)
        # Define the energy terms
        dissipated_energy = (
            Gc
            / cw
            * (
                self.w(alpha) / ell
                + ell * ufl.dot(ufl.grad(alpha), A * ufl.grad(alpha))
            )
            * dx
        )
        # Define the total energy
        return dissipated_energy

    def energy(self, state, domain):
        """
        Compute the energy.

        Parameters
        ----------
        state : dict
            Dictionary containing state variables.
        domain : Domain
            The domain object representing the computational domain.

        Returns
        -------
        ufl.form.Expression
            Total energy.
        """
        # Define the energy terms
        elastic_energy = self.elastic_energy(state, domain)
        dissipated_energy = self.fracture_dissipation(state, domain)
        # Define the total energy
        return elastic_energy + dissipated_energy
