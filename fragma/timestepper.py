class ProportionalTimeStepper:
    def __init__(self, dt):
        # Initialize the time
        self.t = 0
        # Set the initial time step
        self.dt = dt

    def increment(self):
        # Increment time
        self.t += self.dt
