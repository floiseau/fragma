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
from endchecker import choose_end_checker


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
    end_checker : EndChecker
        End checker to checker if the simulation must end.
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
        # Define the model
        self.define_model(self.domain)
        # Define subproblems
        self.subproblems = {}
        self.define_subproblems()
        # Check if path-following is used
        self.use_path_following = (
            pars.get("loading", {}).get("constraint", None) is not None
        )
        # Initialize post-processing
        postprocess_pars = pars.get("postprocess", {})
        self.postprocessor = PostProcessor(
            self.domain, self.model, self.state, postprocess_pars
        )
        # Initialize the exporter
        functions_to_export = list(self.state.values()) + list(
            self.postprocessor.funcs.values()
        )
        scalar_data = self.postprocessor.scalar_data
        probes = self.postprocessor.probes
        self.exporter = Exporter(
            self.domain.mesh, functions_to_export, scalar_data, probes
        )
        # Initialize the time stepper
        self.stepper = ProportionalTimeStepper()
        # Initialize the end checker
        end_pars = self.pars["end"]
        self.end_checker = choose_end_checker(
            end_pars, self.stepper, self.postprocessor
        )

    def define_state_variables(self):
        """
        Define the state variables for the problem.

        This method must be implemented in the child class.
        """
        raise NotImplementedError(
            "Solver: The method 'define_state_variables' must be implemented in the child class."
        )

    def define_model(self, domain):
        """
        Define the model for the problem.

        This method must be implemented in the child class.
        """
        raise NotImplementedError(
            "Solver: The method 'define_model' must be implemented in the child class."
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
        while not self.end_checker.end():
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
