from typing import Callable

import biorbd_casadi as biorbd
from casadi import MX, DM, sin, cos, inv
import numpy as np

from ..misc.utils import check_version
from ..misc.mapping import BiMappingList
from .biorbd_model import BiorbdModel
from .stochastic_bio_model import StochasticBioModel, NoiseType

check_version(biorbd, "1.9.9", "1.10.0")


class StochasticBiorbdModel(BiorbdModel):
    """
    This class allows to define a biorbd model.
    """

    def __init__(
        self,
        bio_model: str | BiorbdModel,
        n_references: int,
        sensory_noise_magnitude: np.ndarray | DM,
        motor_noise_magnitude: np.ndarray | DM,
        sensory_reference: Callable,
        motor_noise_mapping: BiMappingList = BiMappingList(),
        n_collocation_points: int = 1,
        force_field_magnitude: float = 0,
        **kwargs,
    ):
        super().__init__(bio_model if isinstance(bio_model, str) else bio_model.model, **kwargs)

        self.motor_noise_magnitude = motor_noise_magnitude
        self.sensory_noise_magnitude = sensory_noise_magnitude

        self.sensory_reference = sensory_reference

        self.motor_noise_sym = MX.sym("motor_noise", motor_noise_magnitude.shape[0])
        self.sensory_noise_sym = MX.sym("sensory_noise", sensory_noise_magnitude.shape[0])
        self.motor_noise_mapping = motor_noise_mapping
        self.force_field_magnitude = force_field_magnitude

        # TODO: this should be changed when other dynamics are implemented
        n_noised_states = 6 # self.nb_q - self.nb_root
        n_noise = motor_noise_magnitude.shape[0] + sensory_noise_magnitude.shape[0]

        n_noised_controls = self.nb_tau
        if motor_noise_mapping is not None and "tau" in motor_noise_mapping:
            n_noised_controls = len(motor_noise_mapping["tau"].to_second.map_idx)

        self.matrix_shape_k = (n_noised_controls, n_references)
        self.matrix_shape_c = (n_noised_states, n_noise)
        self.matrix_shape_a = (n_noised_states, n_noised_states)
        self.matrix_shape_cov = (n_noised_states, n_noised_states)
        self.matrix_shape_cov_cholesky = (n_noised_states, n_noised_states)
        self.matrix_shape_m = (n_noised_states, n_noised_states * n_collocation_points)

    def stochastic_dynamics(self, q, qdot, tau, ref, k, noise_type: NoiseType, with_gains=True):
        """ note: it should be called feedback dynamics ?? """

        k_matrix = StochasticBioModel.reshape_sym_to_matrix(k, self.matrix_shape_k)

        if noise_type == NoiseType.NONE:
            motor_noise = 0
            sensory_noise = 0
        elif noise_type == NoiseType.MAGNITUDE:
            motor_noise = self.motor_noise_magnitude
            sensory_noise = self.sensory_noise_magnitude
        elif noise_type == NoiseType.SYMBOLIC:
            motor_noise = self.motor_noise_sym
            sensory_noise = self.sensory_noise_sym
        else:
            ValueError("Wrong noise_type")

        tau_fb = tau

        if with_gains:
            end_effector = self.sensory_reference(self, q, qdot)
            tau_fb += self._get_excitation_with_feedback(k_matrix, end_effector, ref, sensory_noise)

        tau_force_field = self._get_force_field(q)

        torques_computed = tau_fb + motor_noise + tau_force_field

        mass_matrix = self.mass_matrix(q)
        non_linear_effects = self.non_linear_effects(q, qdot)

        return inv(mass_matrix) @ (torques_computed - non_linear_effects - self.friction_coefficients @ qdot)

    @staticmethod
    def _get_excitation_with_feedback(k_matrix, computed, ref, sensory_noise):
        """
        Get the effect of the feedback.

        Parameters
        ----------
        k_matrix: MX.sym
            The feedback gains
        computed: MX.sym
            The computed ref and velocity of the hand
        ref: MX.sym
            The reference position and velocity of the hand
        sensory_noise: MX.sym
            The sensory noise
        """

        return k_matrix @ ((computed - ref) + sensory_noise)

    def _get_force_field(self, q):
        """
        Get the effect of the force field.

        Parameters
        ----------
        q: MX.sym
            The generalized coordinates
        """

        l1 = 0.3
        l2 = 0.33
        f_force_field: MX = self.force_field_magnitude * (l1 * cos(q[0]) + l2 * cos(q[0] + q[1]))
        hand_pos = MX(2, 1)
        hand_pos[0] = l2 * sin(q[0] + q[1]) + l1 * sin(q[0])
        hand_pos[1] = l2 * sin(q[0] + q[1])
        tau_force_field = -f_force_field @ hand_pos
        return tau_force_field
