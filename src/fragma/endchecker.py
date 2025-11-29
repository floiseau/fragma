"""
Module providing utilities for choosing and initializing end checkers for simulations.
"""


def choose_end_checker(end_pars, time_stepper, postprocessor):
    """
    Choose and initialize the end checker.

    Parameters
    ----------
    end_pars : dict
        Parameters used to determine the end of the simulation.
    time_stepper : ProportionalTimeStepper
        Time stepper for time integration during simulation.
    postprocessor : PostProcessor
        Post-processor for analyzing simulation results.

    Returns
    -------
    EndChecker
        The initialized end checker.

    Raises
    ------
    RuntimeError
        If the specified end criterion does not exist.
    """
    match end_pars["criterion"]:
        case "t":
            t_max = end_pars["t_max"]
            return TimeEndChecker(time_stepper, t_max)
        case "elastic_energy_drop":
            drop = end_pars["drop"]
            return ElasticEnergyDropEndChecker(postprocessor, drop)
        case _:
            raise RuntimeError(
                f"The end criterion '{end_pars['criterion']}' does not exists."
            )


class TimeEndChecker:
    """
    Class for checking if the end of the simulation is reached based on time.
    With this end checker, the end of the simulation is reached when the time reaches 1.

    Attributes
    ----------
    stepper : ProportionalTimeStepper
        Time stepper for time integration during simulation.
    t_max : int
        Final time step.
    """

    def __init__(self, time_stepper, t_max):
        """
        Initialize the EndChecker.

        Parameters
        ----------
        time_stepper : ProportionalTimeStepper
            Time stepper for time integration during simulation.
        t_max : int
            Final time step.
        """
        # Store the time stepper
        self.time_stepper = time_stepper
        # Store the final time step
        self.t_max = t_max

    def end(self):
        """
        Check if the end time of the simulation is reached.

        Returns
        -------
        bool
            True if the end time of the simulation is reached, False otherwise.
        """
        return self.time_stepper.t > self.t_max


class ElasticEnergyDropEndChecker:
    """
    Class for checking if the end of the simulation is reached based on elastic energy drop.
    With this end checker, the end of the simulation is reached when elastic energy reaches drop % of its maximum value.

    Attributes
    ----------
    postprocessor : PostProcessor
        Post-processor for analyzing simulation results.
    maximum_elastic_energy : float
        Maximum elastic energy encountered during the simulation.
    drop : float
        Drop coefficient.
    """

    def __init__(self, postprocessor, drop):
        """
        Initialize the ElasticEnergyEndChecker.

        Parameters
        ----------
        postprocessor : PostProcessor
            Post-processor for analyzing simulation results.
        drop : float
            Drop coefficient.
        """
        # Store the post processor
        self.postprocessor = postprocessor
        # Get the drop coefficient
        self.drop = drop
        # Initiliaze the maximum elastic energy
        self.maximum_elastic_energy = 0

    def end(self):
        """
        Check if the end time of the simulation is reached based on elastic energy drop.

        Returns
        -------
        bool
            True if the end time of the simulation is reached, False otherwise.
        """
        # Get the current elastic energy
        current_elastic_energy = self.postprocessor.scalar_data["elastic_energy"]
        # Update the maximum elastic energy
        self.maximum_elastic_energy = max(
            self.maximum_elastic_energy, current_elastic_energy
        )
        # Check if the current elastic energy is less than 1% of the maximum elastic energy
        return current_elastic_energy < self.drop * self.maximum_elastic_energy
