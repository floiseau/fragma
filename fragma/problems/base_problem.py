import json

from domain import Domain
from exporter import Exporter
from postprocess import PostProcessor


class BaseProblem:
    def __init__(self, pars):
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
        raise NotImplementedError(
            "Solver: The method 'define_state_variables' must be implemented in the child class."
        )

    def define_subproblems(self):
        raise NotImplementedError(
            "Solver: The method 'define_subproblems' must be implemented in the child class."
        )

    def update_subproblems(self, t: float):
        for subproblem in self.subproblems.values():
            subproblem.update(t)

    def solve(self):
        print("\n████ RESOLUTION")
        # Start the loading iterations
        t_max = self.pars["loading"]["t_max"]
        for t in range(t_max + 1):
            # Display information
            print(f"== Load step {t}/{t_max}")
            # Update subproblems
            self.update_subproblems(t)
            # Solve the problems for this iteration
            self.solve_iteration()
            # Apply post processing
            self.postprocessor.postprocess()
            # Export the results
            self.exporter.export(t)
        # End export
        self.exporter.end()

    def solve_iteration(self):
        raise NotImplementedError(
            "Solver: The method 'solve_iteration' must be implemented in the child class."
        )
