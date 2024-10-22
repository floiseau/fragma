import numpy as np

import ufl

from utils.parameter_parser import parse_parameter


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
            self.dT = parse_parameter(self.thermal_load["dT"], domain)

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
        dx = ufl.Measure("dx", domain=domain.mesh)
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
        if self.dis_model in ["Foc2", "Foc4"]:
            self.tau = parse_parameter(pars["mechanical"]["tau"], domain)
            self.omega = parse_parameter(pars["mechanical"]["omega"], domain)

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
            case "AT" | "Foc2":
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
            case "AT1":
                return alpha
            case "AT2" | "Foc2":
                return alpha**2
            case "DW":
                return 16 * alpha**2 * (1 - alpha) ** 2
            case "Foc4":
                bw = 2 ** (-4 / 3)
                return 3 / bw * alpha**4
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
            case "AT1":
                return 8 / 3
            case "AT2" | "Foc2":
                return 2
            case "DW":
                return 4 * 2 / 3
            case "Foc4":
                return 4
            case _:
                raise ValueError(
                    f"The degradation model named '{self.dis_model}' does not exists."
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
        dx = ufl.Measure("dx", domain=domain.mesh)
        # Get state variables
        alpha = state["alpha"]
        # Get the fracture parameters
        Gc = self.Gc
        ell = self.ell
        cw = self.cw()
        # Check the model
        match self.dis_model:
            case "Foc2":
                # TODO Replace the anisotropy tensor representation by the harmonic decomposition (or another tensor decomposition depending on the indicial symmetries !!!)
                # Parameters
                omega = self.omega
                tau = self.tau
                # Define the anistropy tensor
                id2 = ufl.Identity(2)
                D = ufl.as_tensor(
                    [
                        [ufl.cos(2 * omega), ufl.sin(2 * omega)],
                        [ufl.sin(2 * omega), -ufl.cos(2 * omega)],
                    ]
                )
                B = id2 - tau * D
                # Define the anisotropy function
                grada = ufl.grad(alpha)
                grad2 = ufl.outer(grada, grada)
                phi2 = ufl.inner(B, grad2)
                # Define the dissipation terms
                dissipated_energy = Gc / cw * (self.w(alpha) / ell + ell * phi2) * dx
            case "Foc4":
                # TODO Replace the anisotropy tensor representation by the harmonic decomposition (or another tensor decomposition depending on the indicial symmetries !!!)
                # TODO Replace the anisotropy tensor representation by the harmonic decomposition (or another tensor decomposition depending on the indicial symmetries !!!)
                # TODO Replace the anisotropy tensor representation by the harmonic decomposition (or another tensor decomposition depending on the indicial symmetries !!!)
                # TODO Replace the anisotropy tensor representation by the harmonic decomposition (or another tensor decomposition depending on the indicial symmetries !!!)
                # TODO Replace the anisotropy tensor representation by the harmonic decomposition (or another tensor decomposition depending on the indicial symmetries !!!)
                # Parameters
                omega = self.omega
                tau = self.tau
                # Define the anistropy tensor
                id2 = ufl.Identity(2)
                D_np = np.empty((2, 2, 2, 2))
                D_np[0, 0, 0, 0] = -ufl.cos(4 * omega)
                D_np[1, 1, 1, 1] = D_np[0, 0, 0, 0]
                D_np[0, 0, 1, 1] = ufl.cos(4 * omega)
                D_np[1, 1, 0, 0] = D_np[0, 0, 1, 1]
                D_np[0, 1, 0, 1] = D_np[0, 0, 1, 1]
                D_np[1, 0, 1, 0] = D_np[0, 0, 1, 1]
                D_np[0, 1, 1, 0] = D_np[0, 0, 1, 1]
                D_np[1, 0, 0, 1] = D_np[0, 0, 1, 1]
                D_np[0, 0, 0, 1] = -ufl.sin(4 * omega)
                D_np[0, 0, 1, 0] = D_np[0, 0, 0, 1]
                D_np[0, 1, 0, 0] = D_np[0, 0, 0, 1]
                D_np[1, 0, 0, 0] = D_np[0, 0, 0, 1]
                D_np[1, 1, 1, 0] = ufl.sin(4 * omega)
                D_np[1, 1, 0, 1] = D_np[1, 1, 1, 0]
                D_np[1, 0, 1, 1] = D_np[1, 1, 1, 0]
                D_np[0, 1, 1, 1] = D_np[1, 1, 1, 0]
                D = ufl.as_tensor(D_np)
                B = ufl.outer(id2, id2) - tau * D
                # Define the anisotropy function
                grada = ufl.grad(alpha)
                grad2 = ufl.outer(grada, grada)
                grad4 = ufl.outer(grad2, grad2)
                phi4 = ufl.inner(B, grad4)
                # Define the dissipation terms
                dissipated_energy = Gc / cw * (self.w(alpha) / ell + ell**3 * phi4) * dx
            case _:
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
