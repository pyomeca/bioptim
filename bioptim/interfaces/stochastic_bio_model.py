from casadi import MX

from .biomodel import BioModel


class StochasticBioModel(BioModel):
    """
    This class allows to define a model that can be used in a stochastic optimal control problem.
    """

    sensory_noise_magnitude: float
    motor_noise_magnitude: float

    sensory_noise_sym: MX.sym
    motor_noise_sym: MX.sym

    def stochastic_dynamics(self, q, qdot, tau, ref, k, symbolic_noise=False, with_gains=True):
        """The stochastic dynamics that should be applied to the model"""