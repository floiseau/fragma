class ProportionalTimeStepper:
    """A simple time stepper with proportional time increments.

    This class represents a simple time stepper with proportional time increments.
    It increments the time by a fixed time step in each step.

    Attributes:
        t (float): The current time.
    """

    def __init__(self):
        """Initialize the time stepper."""
        # Initialize the time
        self.t = 0

    def increment(self):
        """Increment the time by the time step."""
        # Increment time
        self.t += 1
