"""
Base Problem Module
===================

This module provides a base class for defining and solving problems.

Classes:
    BaseProblem: Base class for defining and solving problems.
"""

import json

from domain import Domain
from steppers import ProportionalTimeStepper
from exporter import Exporter
from postprocess import PostProcessor


class BaseProblem:
    """
    Base class for defining and solving problems.

    Attributes
    ----------
    pars : dict
        Dictionary containing parameters for the problem.
    domain : Domain
        The domain over which the problem is defined.
    subproblems : dict
        Dictionary containing subproblems of the main problem.
    postprocessor : PostProcessor
        Post-processor for analyzing simulation results.
    exporter : Exporter
        Exporter for saving simulation results.
    stepper : ProportionalTimeStepper
        Time stepper for time integration during simulation.
    """

    def __init__(self, pars):
        """
        Initialize the BaseProblem.

        Parameters
        ----------
        pars : dict
            Dictionary containing parameters for the problem.
        """
        ### Parameters
        print("\n████ PARAMETERS")
        # Store paramters
        self.pars = pars
        # Display a summary
        print(json.dumps(self.pars, indent=4))
        # Define the domain
        self.domain = Domain(pars["mesh"], pars["model"]["dim"])
        # Define the state variables
        self.define_state_variables()
        # Define subproblems
        self.subproblems = {}
        self.define_subproblems()
        # Initialize post-processing
        postprocess_pars = pars.get("postprocess", {})
        self.postprocessor = PostProcessor(
            self.domain, self.model, self.state, postprocess_pars
        )
        # Initialize the exporter
        functions_to_export = list(self.state.values()) + list(
            self.postprocessor.funcs.values()
        )
        probes = self.postprocessor.probes
        self.exporter = Exporter(self.domain.mesh, functions_to_export, probes)

    def define_state_variables(self):
        """
        Define the state variables for the problem.

        This method must be implemented in the child class.
        """
        raise NotImplementedError(
            "Solver: The method 'define_state_variables' must be implemented in the child class."
        )

    def define_subproblems(self):
        """
        Define the subproblems for the problem.

        This method must be implemented in the child class.
        """
        raise NotImplementedError(
            "Solver: The method 'define_subproblems' must be implemented in the child class."
        )

    def update_subproblems(self, t: float):
        """
        Update the subproblems for the current time step.

        Parameters
        ----------
        t : float
            Current time.
        """
        for subproblem in self.subproblems.values():
            subproblem.update(t)

    def solve(self):
        """
        Solve the problem over time.
        """
        print("\n████ RESOLUTION")
        # Initialize the time stepper
        self.stepper = ProportionalTimeStepper(self.pars["loading"]["dt"])
        while self.stepper.not_end():
            # Get time
            t = self.stepper.t
            # Display information
            print(f"\n== Time {t:.8g}")
            # Update subproblems
            self.update_subproblems(t)
            # Solve the problems for this iteration
            self.solve_iteration()
            # Apply post processing
            self.postprocessor.postprocess()
            # Export the results
            self.exporter.export(t)
            # Increment the time stepper
            self.stepper.increment()
        # End export
        self.exporter.end()

    def solve_iteration(self):
        """
        Solve a single iteration of the problem.

        This method must be implemented in the child class.
        """
        raise NotImplementedError(
            "Solver: The method 'solve_iteration' must be implemented in the child class."
        )
