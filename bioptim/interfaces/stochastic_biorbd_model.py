import biorbd_casadi as biorbd
from casadi import MX, vertcat

from ..misc.utils import check_version
from .biorbd_model import BiorbdModel

check_version(biorbd, "1.9.9", "1.10.0")


class StochasticBiorbdModel(BiorbdModel):
    """
    This class allows to define a biorbd model.
    """

    def __init__(self, bio_model: str | BiorbdModel,
                 sensory_noise_magnitude,
                 motor_noise_magnitude):

        super().__init__(bio_model if isinstance(bio_model, str) else bio_model.model)

        self.sensory_noise_sym = MX.sym("sensory_noise", self.nb_q + self.nb_qdot)
        self.motor_noise_sym = MX.sym("motor_noise", self.nb_tau)
        self.motor_noise_magnitude = motor_noise_magnitude
        self.sensory_noise_magnitude = sensory_noise_magnitude
        # self.friction = np.array([[0.05, 0.025], [0.025, 0.05]])

    def stochastic_dynamics(self, q, qdot, tau, ref, k, symbolic_noise=False, with_gains=True):
        """ note: it should be called feedback dynamics ?? """

        from bioptim.optimization.optimization_variable import OptimizationVariable  # Avoid circular import
        k_matrix = OptimizationVariable.reshape_sym_to_matrix(k, self.nb_tau, self.nb_q + self.nb_qdot)

        motor_noise = self.motor_noise_sym if symbolic_noise else 0
        sensory_noise = self.sensory_noise_sym if symbolic_noise else 0

        tau_fb = tau

        if with_gains:
            hand_pos = self.markers(q)[2][:2]
            hand_vel = self.marker_velocities(q, qdot)[2][:2]
            end_effector = vertcat(hand_pos, hand_vel)

            tau_fb += self._get_excitation_with_feedback(k_matrix, end_effector, ref, sensory_noise)

        tau_force_field = self._get_force_field(q)

        torques_computed = tau_fb + motor_noise + tau_force_field

        mass_matrix = self.mass_matrix(q)
        non_linear_effects = self.non_linear_effects(q, qdot)

        return inv(mass_matrix) @ (torques_computed - non_linear_effects - self.friction @ qdot)

    @staticmethod
    def _get_excitation_with_feedback(k_matrix, hand_pos_velo, ref, sensory_noise):
        """
        Get the effect of the feedback.
        Parameters
        ----------
        k_matrix: MX.sym
            The feedback gains
        hand_pos_velo: MX.sym
            The position and velocity of the hand
        ref: MX.sym
            The reference position and velocity of the hand
        sensory_noise: MX.sym
            The sensory noise
        """

        return k_matrix @ ((hand_pos_velo - ref) + sensory_noise)

    # def _get_force_field(self, q):
    #     """
    #     Get the effect of the force field.
    #     Parameters
    #     ----------
    #     q: MX.sym
    #         The generalized coordinates
    #     """
    #
    #     l1 = 0.3
    #     l2 = 0.33
    #     f_force_field: MX = self.force_field_magnitude * (l1 * cos(q[0]) + l2 * cos(q[0] + q[1]))
    #     hand_pos = MX(2, 1)
    #     hand_pos[0] = l2 * sin(q[0] + q[1]) + l1 * sin(q[0])
    #     hand_pos[1] = l2 * sin(q[0] + q[1])
    #     tau_force_field = -f_force_field @ hand_pos
    #     return tau_force_field