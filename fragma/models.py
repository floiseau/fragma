import numpy as np

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

    def __init__(self, pars):
        """
        Initialize the BaseModel.

        Parameters
        ----------
        pars : dict
            Dictionary containing parameters of the material model.
        """
        # Get elastic parameters
        E = pars["mechanical"]["E"]
        nu = pars["mechanical"]["nu"]
        # Compute Lame coefficient
        self.la = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))
        # Check the 2D assumption
        if pars["model"]["dim"] == 2:
            assumption = pars["model"]["2D_assumption"]
            match assumption:
                case "plane_stress":
                    print("Plane stress assumption")
                    self.la = 2 * self.mu * self.la / (self.la + 2 * self.mu)
                case "plane_strain":
                    print("Plane strain assumption")
                case _:
                    raise ValueError(f'The 2D assumption "{assumption}" in unknown')

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

    def __init__(self, pars):
        """
        Initialize the ElasticModel.

        Parameters
        ----------
        pars : dict
            Dictionary containing parameters of the material model.
        """
        # Initialise parent class
        super().__init__(pars)

    def energy(self, state, domain):
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
        # Get the mesh
        mesh = domain.mesh
        # Get the dimension of the mesh
        dim = mesh.geometry.dim
        # Get the integrands
        dx = ufl.Measure("dx", domain=mesh)
        ds = ufl.Measure("ds", domain=mesh)
        # Define the imposed stress on the remaining of the boundary
        T = fem.Constant(mesh, default_scalar_type([0 for d in range(dim)]))
        # Define the volumic forces
        f = fem.Constant(mesh, default_scalar_type([0 for d in range(dim)]))
        # Get state variables
        u = state["u"]
        # Define the energy terms
        elastic_energy = 0.5 * ufl.inner(self.sig(state), self.eps(state)) * dx
        external_work = ufl.dot(f, u) * dx + ufl.dot(T, u) * ds
        # Define the total energy
        return elastic_energy - external_work


class FractureModel(BaseModel):
    """
    Material model for fracture mechanics.

    This class implements the material model for fracture mechanics.

    Parameters
    ----------
    pars : dict
        Dictionary containing parameters of the material model.
    """

    def __init__(self, pars):
        """
        Initialize the FractureModel.

        Parameters
        ----------
        pars : dict
            Dictionary containing parameters of the material model.
        """
        # Initialise parent class
        super().__init__(pars)
        # Get the degradation model
        self.deg_model = pars["model"]["model"]
        # Get fracture parameters
        self.Gc = pars["mechanical"]["Gc"]
        self.ell = pars["mechanical"]["ell"]
        self.aG = pars["mechanical"]["aG"] if "aG" in pars["mechanical"] else 0
        self.theta_0 = (
            pars["mechanical"]["theta_0"] if "theta_0" in pars["mechanical"] else 0
        )
        # Get the residual crack phase
        self.alpha_res = pars["numerical"]["alpha_res"]

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
            case "AT1":
                return (1 - alpha) ** 2 + alpha_res
            case "AT2":
                return (1 - alpha) ** 2 + alpha_res
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
            case "AT1":
                return -2 * (1 - alpha)
            case "AT2":
                return -2 * (1 - alpha)
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
        match self.deg_model:
            case "AT1":
                return alpha
            case "AT2":
                return alpha**2
            case _:
                raise ValueError(
                    f"The degradation model named '{self.deg_model}' does not exists."
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
        match self.deg_model:
            case "AT1":
                return 1
            case "AT2":
                return 2 * alpha
            case _:
                raise ValueError(
                    f"The degradation model named '{self.deg_model}' does not exists."
                )

    def cw(self):
        """
        Normalization coefficient.

        Returns
        -------
        float
            Normalization coefficient.
        """
        match self.deg_model:
            case "AT1":
                return 8 / 3
            case "AT2":
                return 2
            case _:
                raise ValueError(
                    f"The degradation model named '{self.deg_model}' does not exists."
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
        # Get the mesh from the domain
        mesh = domain.mesh
        # Get the dimension of the mesh
        dim = mesh.geometry.dim
        # Get the integrands
        dx = ufl.Measure("dx", domain=mesh)
        ds = ufl.Measure("ds", domain=mesh)
        # Define the imposed stress on the remaining of the boundary
        T = fem.Constant(mesh, default_scalar_type([0 for d in range(dim)]))
        # Define the volumic forces
        f = fem.Constant(mesh, default_scalar_type([0 for d in range(dim)]))
        # Get state variables
        u, alpha = state["u"], state["alpha"]
        # Get the fracture parameters
        Gc, ell = self.Gc, self.ell
        cw = self.cw()
        # Compute the anisotropy matrix
        aG, theta_0 = self.aG, self.theta_0
        A_np = np.eye(dim)
        if aG != 0:
            A_np += aG * np.array(
                [
                    [np.cos(2 * theta_0), np.sin(2 * theta_0)],
                    [np.sin(2 * theta_0), -np.cos(2 * theta_0)],
                ]
            )
        A = fem.Constant(mesh, A_np)
        # Define the energy terms
        elastic_energy = 0.5 * ufl.inner(self.sig_eff(state), self.eps(state)) * dx
        dissipated_energy = (
            Gc
            / cw
            * (
                self.w(alpha) / ell
                + ell * ufl.dot(ufl.grad(alpha), A * ufl.grad(alpha))
            )
            * dx
        )
        external_work = ufl.dot(f, u) * dx + ufl.dot(T, u) * ds
        # Define the total energy
        return elastic_energy + dissipated_energy - external_work
