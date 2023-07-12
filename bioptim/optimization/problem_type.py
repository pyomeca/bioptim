from casadi import DM


class OcpType:
    """
    Selection of the type of optimization problem to be solved.
    """

    def __init__(self):
        pass

    class SOCP_EXPLICIT:
        """
        The class used to declare a stochastic problem with explicit stochastic dynamics
        Attributes
        ----------
        wM_magnitude: DM
            The magnitude of the motor noise
        wS_magnitude: DM
            The magnitude of the sensory noise
        """

        def __init__(self, wM_magnitude: DM, wS_magnitude: DM):
            self.wM_magnitude = wM_magnitude
            self.wS_magnitude = wS_magnitude

    class SOCP_IMPLICIT:
        """
        The class used to declare a stochastic problem with implicit stochastic dynamics
        Attributes
        ----------
        wM_magnitude: DM
            The magnitude of the motor noise
        wS_magnitude: DM
            The magnitude of the sensory noise
        """

        def __init__(self, wM_magnitude: DM, wS_magnitude: DM):
            self.wM_magnitude = wM_magnitude
            self.wS_magnitude = wS_magnitude
