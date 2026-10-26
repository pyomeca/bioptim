from ..configure_variables import States, Controls
from .abstract_dynamics import StateDynamics


class VariationalTorqueDynamics(StateDynamics):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # If the model has no holonomic constraint, there will be no lambdas defined

    @property
    def state_configuration_functions(self):
        return [States.Q, States.LAMBDA]

    @property
    def control_configuration_functions(self):
        return [Controls.TAU]

    @property
    def algebraic_configuration_functions(self):
        return []

    @property
    def extra_configuration_functions(self):
        return []

    @property
    def has_continuous_dynamics(self):
        return False

    def dynamics(self, *args, **kwargs):
        raise NotImplementedError(
            "The variational integrator has no continuous-time dynamics, the discrete Euler-Lagrange equations are "
            "declared by OdeSolver.VARIATIONAL."
        )

    @property
    def extra_dynamics(self):
        return None
