import numpy as np

from dolfinx import fem, default_scalar_type
import ufl


class BaseModel:
    def __init__(self, pars):
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
        return ufl.sym(ufl.grad(state["u"]))

    def sig(self, state):
        # Get elastic parameters
        mu, la = self.mu, self.la
        # Get the state variables
        u = state["u"]
        # Compute the stess
        return la * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0 * mu * self.eps(state)

    def energy(self, state, mesh):
        raise NotImplementedError(
            "Model: The method 'energy' must be implemented in the child class."
        )


class ElasticModel(BaseModel):
    """TODO"""

    def __init__(self, pars):
        # Initialise parent class
        super().__init__(pars)

    def energy(self, state, domain):
        # Get the dimension of the domain
        dim = domain.geometry.dim
        # Get the integrands
        dx = ufl.Measure("dx", domain=domain)
        ds = ufl.Measure("ds", domain=domain)
        # Define the imposed stress on the remaining of the boundary
        T = fem.Constant(domain, default_scalar_type([0 for d in range(dim)]))
        # Define the volumic forces
        f = fem.Constant(domain, default_scalar_type([0 for d in range(dim)]))
        # Get state variables
        u = state["u"]
        # Define the energy terms
        elastic_energy = 0.5 * ufl.inner(self.sig(state), self.eps(state)) * dx
        external_work = ufl.dot(f, u) * dx + ufl.dot(T, u) * ds
        # Define the total energy
        return elastic_energy - external_work


class FractureModel(BaseModel):
    """TODO"""

    def __init__(self, pars):
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

    def w(self, alpha):
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

    def cw(self):
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
        return self.a(state["alpha"]) * self.sig(state)

    def energy(self, state, domain):
        # Get the dimension of the domain
        dim = domain.geometry.dim
        # Get the integrands
        dx = ufl.Measure("dx", domain=domain)
        ds = ufl.Measure("ds", domain=domain)
        # Define the imposed stress on the remaining of the boundary
        T = fem.Constant(domain, default_scalar_type([0 for d in range(dim)]))
        # Define the volumic forces
        f = fem.Constant(domain, default_scalar_type([0 for d in range(dim)]))
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
        A = fem.Constant(domain, A_np)
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


class FractureModelMiehe(FractureModel):
    """Model associated to the solver of Miehe et at. (2010).

    The only different the FractureModel is the introduction of the history field and its impact on the energy.
    """

    def update_history(self, state):
        # Get the state variable
        u, H = state["u"], state["H"]
        # Get the dim
        D = len(u)

        ### Compute Phi0
        # Get the elastic paramaters
        mu, la = self.mu, self.la
        # Get the function space of H
        V_H = H.function_space
        # Compute eps
        eps = ufl.sym(ufl.grad(state["u"]))
        # Compute its hydrostatic and deviatoric parts
        tr_eps = ufl.inner(eps, ufl.Identity(D))
        eps_dev = eps - 1 / D * tr_eps * ufl.Identity(D)
        # Compute Phi0
        # WARNING: Is the positive part of tr_eps necessary ????
        Phi0_ufl = (
            1 / 2 * ((la + 2 * mu / D) * tr_eps**2 + mu * ufl.inner(eps_dev, eps_dev))
        )
        # Generate the FEM expression
        Phi0_expr = fem.Expression(Phi0_ufl, V_H.element.interpolation_points())
        # Generate the function and interpolate it
        Phi0 = fem.Function(V_H)
        Phi0.interpolate(Phi0_expr)
        # Compute H
        H.vector[:] = np.maximum(Phi0.vector[:], H.vector[:])
        H.vector.assemble()

    def energy(self, state, domain):
        # Get the dimension of the domain
        dim = domain.geometry.dim
        # Get the integrands
        dx = ufl.Measure("dx", domain=domain)
        ds = ufl.Measure("ds", domain=domain)
        # Define the imposed stress on the remaining of the boundary
        T = fem.Constant(domain, default_scalar_type([0 for d in range(dim)]))
        # Define the volumic forces
        f = fem.Constant(domain, default_scalar_type([0 for d in range(dim)]))
        # Get state variables
        u, alpha, H = state["u"], state["alpha"], state["H"]
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
        A = fem.Constant(domain, A_np)
        # Define the energy terms
        elastic_energy = 0.5 * ufl.inner(self.sig_eff(state), self.eps(state)) * dx
        dissipated_energy = (
            Gc
            / cw
            * (
                self.w(alpha) / ell
                + ell * ufl.dot(ufl.grad(alpha), A * ufl.grad(alpha))
            )
            - H * (default_scalar_type(1.0) - alpha) ** 2
        ) * dx
        external_work = ufl.dot(f, u) * dx + ufl.dot(T, u) * ds
        # Define the total energy
        return elastic_energy + dissipated_energy - external_work
