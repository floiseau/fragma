class ProportionalTimeStepper:
    """A simple time stepper with proportional time increments.

    This class represents a simple time stepper with proportional time increments.
    It increments the time by a fixed time step in each step.

    Parameters:
        dt (float): The initial time step.

    Attributes:
        t (float): The current time.
        dt (float): The time step.

    """

    def __init__(self, dt: float):
        """Initialize the time stepper.

        Args:
            dt (float): The initial time step.

        """
        # Initialize the time
        self.t = 0
        # Set the initial time step
        self.dt = dt

    def increment(self):
        """Increment the time by the time step."""
        # Increment time
        self.t += self.dt

    def not_end(self):
        return self.t <= 1.0 + 1e-12


# class LoadStepper:
#
#     def __init__(self, dt: float):
#         """Initialize the time stepper.
#
#         Args:
#             dt (float): The initial time step.
#
#         """
#         # Initialize the time
#         self.t = 0
#         # Set the initial time step
#         self.dt = dt
#
#     def increment(self):
#         """Increment the time by the time step."""
#         # Increment time
#         self.t += self.dt
#
#     def not_end(self):
#         return self.t<1.0
