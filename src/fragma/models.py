import itertools

import numpy as np

import ufl

from .utils.parameter_parser import parse_parameter


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
        self.E = parse_parameter(pars["mechanical"]["E"], domain)
        self.nu = parse_parameter(pars["mechanical"]["nu"], domain)
        # Compute Lame coefficient
        self.la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))
        # Check the 2D assumption
        self.dim = pars["model"]["dim"]
        if self.dim == 2:
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
        # Get the optional thermal load (to compute the thermal strain)
        self.thermal_load = pars["loading"].get("thermal_load", {})
        if self.thermal_load:
            # Get the thermal expansion coefficient
            self.a_T = parse_parameter(
                self.thermal_load["thermal_expansion_coeff"], domain
            )
            # Get the temperature field (variation)
            self.dT, self.dT_lambda = parse_parameter(
                self.thermal_load["dT"], domain, export_lambda=True
            )

    def eps_th(self):
        """
        Compute the thermal strain (for thermal loads).

        Returns
        -------
        ufl.form.Expression
            Strain tensor.
        """
        # Compute the thermal strain
        coeff = self.a_T * self.dT if self.thermal_load else 0
        return coeff * ufl.Identity(2)

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

    def eps_ela(self, state):
        """
        Compute the elastic strain tensor.

        Parameters
        ----------
        state : dict
            Dictionary containing state variables.

        Returns
        -------
        ufl.form.Expression
            Strain tensor.
        """
        return self.eps(state) - self.eps_th()

    def ela(self):
        """
        Compute the elasticity tensor.

        Returns
        -------
        ufl.form.Expression
            Elasticity tensor.
        """
        # Define index for tensorial notations
        i, j, k, l = ufl.indices(4)
        # Compute constant tensors
        Id2 = ufl.Identity(self.dim)
        Id2xId2 = ufl.outer(Id2, Id2)
        Id4 = (
            1
            / 2
            * ufl.as_tensor(Id2[i, k] * Id2[j, l] + Id2[i, l] * Id2[j, k], (i, j, k, l))
        )
        # Compute the elasticity tensor
        return 2 * self.mu * Id4 + self.la * Id2xId2

    def ela_eff(self, state):
        """
        Compute the effective elasticity tensor.

        Parameters
        ----------
        state : dict
            Dictionary containing state variables.

        Returns
        -------
        ufl.form.Expression
            Elasticity tensor.
        """
        # Compute the elasticity tensor
        return self.ela()

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
        # Generate indices
        i, j, k, l = ufl.indices(4)
        # Get elastic parameters
        ela = self.ela()
        # Compute the strain
        eps = self.eps(state)
        # Compute the stess
        return ufl.as_tensor(ela[i, j, k, l] * eps[k, l], (i, j))

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
        # Generate indices
        i, j, k, l = ufl.indices(4)
        # Get elastic parameters
        ela_eff = self.ela_eff(state)
        # Compute the strain
        eps = self.eps(state)
        # Compute the stess
        return ufl.as_tensor(ela_eff[i, j, k, l] * eps[k, l], (i, j))

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
        dx = ufl.Measure("dx", domain=domain.mesh, metadata={"quadrature_degree": 12})
        # Compute the effective elasticity tensor
        ela_eff = self.ela_eff(state)
        # Compute the elastic strain
        eps_ela = self.eps_ela(state)
        # Define the total energy
        return 1 / 2 * ufl.inner(ela_eff, ufl.outer(eps_ela, eps_ela)) * dx

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
            self.deg_model = model_par
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
        self.ell = parse_parameter(pars["mechanical"]["ell"], domain)
        # Check for anisotropy
        self.is_anisotropic = "theta_0" in pars["mechanical"]
        if not self.is_anisotropic:
            # Get the critical energy release rate
            self.Gc = parse_parameter(pars["mechanical"]["Gc"], domain)
        else:
            # Get the critical energy release rate (min and max)
            Gc_min = parse_parameter(pars["mechanical"]["Gc_min"], domain)
            Gc_max = parse_parameter(pars["mechanical"]["Gc_max"], domain)
            # Convert to other model parameters
            self.Gc = ufl.sqrt(1 / 2 * (Gc_min**2 + Gc_max**2))
            self.aG = 1 / 2 * (Gc_max**2 - Gc_min**2) / self.Gc**2
            # Ge the anisotropy angle
            self.theta_0 = (
                parse_parameter(pars["mechanical"]["theta_0"], domain) * np.pi / 180
                if "theta_0" in pars["mechanical"]
                else 0
            )
        # Check for model specific parameters
        if self.dis_model in ["Foc2", "Foc4", "RMBRAT1", "RMBRAT2"]:
            self.Gc = parse_parameter(pars["mechanical"]["Gc"], domain)
            self.D2 = parse_parameter(pars["mechanical"]["D2"], domain)
            self.P2 = parse_parameter(pars["mechanical"]["P2"], domain)
        if self.dis_model in ["Foc4", "RMBRAT1", "RMBRAT2"]:
            self.D4 = parse_parameter(pars["mechanical"]["D4"], domain)
            self.P4 = parse_parameter(pars["mechanical"]["P4"], domain)
        if self.dis_model in ["Foc2X"]:
            self.Gc = parse_parameter(pars["mechanical"]["Gc"], domain)
            self.th0 = parse_parameter(pars["mechanical"]["th0"], domain)
            self.dG = parse_parameter(pars["mechanical"]["dG"], domain)

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
            case "AT1" | "AT2" | "Foc2" | "Foc2X" | "RMBR":
                return (1 - alpha) ** 2 + alpha_res
            case "KKL":
                return 4 * (1 - alpha) ** 3 - 4 * (1 - alpha) ** 3 + alpha_res
            case "KSM":
                return 3 * (1 - alpha) ** 2 - 3 * (1 - alpha) ** 2 + alpha_res
            case "Foc4":
                return (1 - alpha**4) ** 2 + alpha_res
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
        # Define a variable
        alpha = ufl.variable(alpha)
        # Comupute the derivative
        return ufl.diff(self.a(alpha), alpha)

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
            case "AT1" | "RMBRAT1":
                return alpha
            case "AT2" | "Foc2" | "Foc2X" | "RMBRAT2":
                return alpha**2
            case "DW":
                return 16 * alpha**2 * (1 - alpha) ** 2
            case "Foc4":
                bw = 2 ** (-4 / 3)
                return 3 / bw * alpha**4
            case _:
                raise ValueError(
                    f"The dissipation model named '{self.dis_model}' does not exists."
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
        # Define a variable
        alpha = ufl.variable(alpha)
        # Comupute the derivative
        return ufl.diff(self.w(alpha), alpha)

    def cw(self):
        """
        Normalization coefficient.

        Returns
        -------
        float
            Normalization coefficient.
        """
        match self.dis_model:
            case "AT1" | "RMBRAT1":
                return 8 / 3
            case "AT2" | "Foc2" | "Foc2X" | "RMBRAT2":
                return 2
            case "DW":
                return 4 * 2 / 3
            case "Foc4":
                return 4
            case _:
                raise ValueError(
                    f"The dissipation model named '{self.dis_model}' does not exists."
                )

    def ela_eff(self, state):
        """
        Compute the effective elasticity tensor.

        Parameters
        ----------
        state : dict
            Dictionary containing state variables.

        Returns
        -------
        ufl.form.Expression
            Elasticity tensor.
        """
        # Compute the elasticity tensor
        return self.a(state["alpha"]) * self.ela()

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
        dx = ufl.Measure("dx", domain=domain.mesh, metadata={"quadrature_degree": 12})
        # Get state variables
        alpha = state["alpha"]
        # Get the fracture parameters
        Gc = self.Gc
        ell = self.ell
        cw = self.cw()
        # Check the model
        match self.dis_model:
            case "Foc2":
                # Define 2-order identity
                id2 = ufl.Identity(2)
                # Define the covariant
                h2_np = np.empty((2, 2))
                h2_np[0, 0] = -self.D2 * ufl.cos(2 * self.P2)  # Idenpendent components
                h2_np[0, 1] = -self.D2 * ufl.sin(2 * self.P2)
                h2_np[1, 1] = -h2_np[0, 0]  # Traceless condition
                h2_np[1, 0] = h2_np[0, 1]  # Symmetry conditions
                h2 = ufl.as_tensor(h2_np)
                B = id2 + h2
                # Define the anisotropy function
                grada = ufl.grad(alpha)
                grad2 = ufl.outer(grada, grada)
                phi2 = ufl.inner(B, grad2)
                # Define the dissipation terms
                dissipated_energy = Gc / cw * (self.w(alpha) / ell + ell * phi2) * dx
            case "Foc2X":
                # Define the anisotropy tensor
                A_np = np.empty((2, 2))
                A_np[0, 0] = -ufl.sin(2 * self.th0)
                A_np[0, 1] = ufl.cos(2 * self.th0)
                A_np[1, 0] = ufl.cos(2 * self.th0)
                A_np[1, 1] = ufl.sin(2 * self.th0)
                A = ufl.as_tensor(A_np)
                # Define the anisotropy function
                grada = ufl.grad(alpha)
                grad2 = ufl.outer(grada, grada)
                phi2_1 = ufl.inner(grada, grada)
                phi2_2 = self.dG / Gc * ufl.inner(A, grad2) ** 2
                phi2 = (phi2_1 + phi2_2) ** 2
                # Define the dissipation term
                dissipated_energy = Gc / cw * (self.w(alpha) / ell + ell * phi2) * dx
            case "Foc4":
                # Define constants
                id2 = ufl.Identity(2)
                id4_np = np.empty((2, 2, 2, 2))
                indices = itertools.product(range(2), range(2), range(2), range(2))
                for i, j, k, l in indices:
                    id4_np[i, j, k, l] = (
                        1 / 2 * (id2[i, k] * id2[j, l] + id2[i, l] * id2[j, k])
                    )
                id4 = ufl.as_tensor(id4_np)
                # Define the 2nd order covariant
                h2_np = np.empty((2, 2))
                h2_np[0, 0] = -self.D2 * ufl.cos(2 * self.P2)  # Idenpendent components
                h2_np[0, 1] = -self.D2 * ufl.sin(2 * self.P2)
                h2_np[1, 1] = -h2_np[0, 0]  # Traceless condition
                h2_np[1, 0] = h2_np[0, 1]  # Symmetry conditions
                h2 = ufl.as_tensor(h2_np)
                # Define the 4th order covariant
                h4_np = np.empty((2, 2, 2, 2))
                h4_np[0, 0, 0, 0] = self.D4 * ufl.cos(
                    4 * self.P4
                )  # Idenpendent components
                h4_np[0, 0, 0, 1] = self.D4 * ufl.sin(4 * self.P4)
                h4_np[0, 0, 1, 1] = -h4_np[0, 0, 0, 0]  # Traceless conditions
                h4_np[0, 1, 1, 1] = -h4_np[0, 0, 0, 1]
                h4_np[1, 1, 1, 1] = h4_np[0, 0, 0, 0]
                h4_np[0, 0, 1, 0] = h4_np[0, 0, 0, 1]  # Symmetries conditions
                h4_np[0, 1, 0, 0] = h4_np[0, 0, 0, 1]
                h4_np[1, 0, 0, 0] = h4_np[0, 0, 0, 1]
                h4_np[1, 1, 0, 0] = h4_np[0, 0, 1, 1]
                h4_np[0, 1, 0, 1] = h4_np[0, 0, 1, 1]
                h4_np[1, 0, 1, 0] = h4_np[0, 0, 1, 1]
                h4_np[1, 0, 0, 1] = h4_np[0, 0, 1, 1]
                h4_np[0, 1, 1, 0] = h4_np[0, 0, 1, 1]
                h4_np[1, 0, 1, 1] = h4_np[0, 1, 1, 1]
                h4_np[1, 1, 0, 1] = h4_np[0, 1, 1, 1]
                h4_np[1, 1, 1, 0] = h4_np[0, 1, 1, 1]
                h4 = ufl.as_tensor(h4_np)
                # Define the anistropy tensor
                B = id4 + 1.0 / 2.0 * (ufl.outer(id2, h2) + ufl.outer(h2, id2)) + h4
                # Compute the crack phase gradient
                grada = ufl.grad(alpha)
                grad2 = ufl.outer(grada, grada)
                grad4 = ufl.outer(grad2, grad2)
                # Compute the surface energy
                phi4 = ufl.inner(B, grad4)
                # Define the dissipation terms
                dissipated_energy = Gc / cw * (self.w(alpha) / ell + ell**3 * phi4) * dx
            case "RMBRAT1" | "RMBRAT2":
                alpha_0 = state["alpha0"]
                grada0 = ufl.grad(alpha_0)
                phi = ufl.conditional(
                    ufl.ge(ufl.sqrt(grada0[0] ** 2 + grada0[1] ** 2), 1e-12),
                    ufl.atan2(-grada0[1], grada0[0]),
                    np.pi / 2,
                )
                theta = phi - np.pi / 2
                Gc = self.Gc * (
                    1
                    + self.D2 * ufl.cos(2 * (theta - self.P2))
                    + self.D4 * ufl.cos(4 * (theta - self.P4))
                ) ** (1 / 4)

                wa = self.w(alpha)
                grada = ufl.grad(alpha)
                grada_grada = ufl.dot(grada, grada)
                dissipated_energy = Gc / cw * (wa / ell + ell * grada_grada) * dx
            case "AT1" | "AT2" | "DW":
                # Compute the anisotropy matrix
                A = ufl.as_tensor(np.eye(domain.mesh.geometry.dim))
                # Add the higher order terms if the model is anisotropic
                if self.is_anisotropic:
                    # Get the parameters
                    aG, theta_0 = self.aG, self.theta_0
                    #  Compute the 2nd order term of the anisotropy tensor
                    A += aG * ufl.as_tensor(
                        np.array(
                            [
                                [np.cos(2 * theta_0), np.sin(2 * theta_0)],
                                [np.sin(2 * theta_0), -np.cos(2 * theta_0)],
                            ]
                        )
                    )
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
            case _:
                raise ValueError(
                    f"The dissipation model {self.dis_model} does not exists."
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
