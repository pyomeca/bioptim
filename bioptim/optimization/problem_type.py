from casadi import DM


class SocpType:
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
        motor_noise_magnitude: DM
            The magnitude of the motor noise
        sensory_noise_magnitude: DM
            The magnitude of the sensory noise
        """

        def __init__(self, motor_noise_magnitude: DM, sensory_noise_magnitude: DM):
            self.motor_noise_magnitude = motor_noise_magnitude
            self.sensory_noise_magnitude = sensory_noise_magnitude

    class SOCP_IMPLICIT:
        """
        The class used to declare a stochastic problem with implicit stochastic dynamics
        Attributes
        ----------
        motor_noise_magnitude: DM
            The magnitude of the motor noise
        sensory_noise_magnitude: DM
            The magnitude of the sensory noise
        """

        def __init__(self, motor_noise_magnitude: DM, sensory_noise_magnitude: DM):
            self.motor_noise_magnitude = motor_noise_magnitude
            self.sensory_noise_magnitude = sensory_noise_magnitude

    class SOCP_COLLOCATION:
        """
        The class used to declare a stochastic problem with implicit stochastic dynamics implemented taking advantage
        of the collocation integration.
        Attributes
        ----------
        motor_noise_magnitude: DM
            The magnitude of the motor noise
        sensory_noise_magnitude: DM
            The magnitude of the sensory noise
        """

        def __init__(self, motor_noise_magnitude: DM, sensory_noise_magnitude: DM):
            self.motor_noise_magnitude = motor_noise_magnitude
            self.sensory_noise_magnitude = sensory_noise_magnitude
