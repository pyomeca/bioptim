from .torque_dynamics_free_floating_base import TorqueFreeFloatingBaseDynamics
from .stochastic_dynamics import StochasticTorqueDynamics


class StochasticTorqueFreeFloatingBaseDynamics(TorqueFreeFloatingBaseDynamics, StochasticTorqueDynamics):
    """
    This class is used to create a model actuated through joint torques with a free floating base with stochastic variables.

    x = [q_roots, q_joints, qdot_roots, qdot_joints]
    u = [tau_joints, stochastic_variables]
    a = [stochastic_variables]
    """

    def __init__(self, problem_type, with_cholesky, n_noised_tau, n_noise, n_noised_states, n_references):
        super().__init__()
        StochasticTorqueDynamics.__init__(
            self, problem_type, with_cholesky, n_noised_tau, n_noise, n_noised_states, n_references
        )
