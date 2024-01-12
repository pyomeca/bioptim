from typing import Callable

from casadi import MX, DM, SX
import numpy as np

from ...misc.mapping import BiMappingList
from bioptim import BiorbdModel, DynamicsFunctions, StochasticBioModel


def _compute_torques_from_noise_and_feedback_default(
    nlp, time, states, controls, parameters, algebraic_states, sensory_noise, motor_noise
):
    tau_nominal = DynamicsFunctions.get(nlp.controls["tau"], controls)

    ref = DynamicsFunctions.get(nlp.algebraic_states["ref"], algebraic_states)
    k = DynamicsFunctions.get(nlp.algebraic_states["k"], algebraic_states)
    k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)

    sensory_input = nlp.model.sensory_reference(time, states, controls, parameters, algebraic_states, nlp)
    tau_fb = k_matrix @ ((sensory_input - ref) + sensory_noise)

    tau_motor_noise = motor_noise

    tau = tau_nominal + tau_fb + tau_motor_noise

    return tau


class StochasticBiorbdModel(BiorbdModel):
    """
    This class allows to define a biorbd model.
    """

    def __init__(
        self,
        bio_model: str | BiorbdModel,
        n_references: int,
        n_feedbacks: int,
        n_noised_states: int,
        n_noised_controls: int,
        sensory_noise_magnitude: np.ndarray | DM,
        motor_noise_magnitude: np.ndarray | DM,
        sensory_reference: Callable,
        compute_torques_from_noise_and_feedback: Callable = _compute_torques_from_noise_and_feedback_default,
        motor_noise_mapping: BiMappingList = BiMappingList(),
        n_collocation_points: int = 1,
        **kwargs,
    ):
        super().__init__(bio_model if isinstance(bio_model, str) else bio_model.model, **kwargs)

        self.motor_noise_magnitude = motor_noise_magnitude
        self.sensory_noise_magnitude = sensory_noise_magnitude

        self.compute_torques_from_noise_and_feedback = compute_torques_from_noise_and_feedback
        
        self.sensory_reference = sensory_reference

        self.motor_noise_mapping = motor_noise_mapping

        self.n_references = n_references
        self.n_feedbacks = n_feedbacks
        self.n_noised_states = n_noised_states
        self.n_noise = motor_noise_magnitude.shape[0] + sensory_noise_magnitude.shape[0]
        self.n_noised_controls = n_noised_controls
        if motor_noise_mapping is not None and "tau" in motor_noise_mapping:
            if self.n_noised_controls != len(motor_noise_mapping["tau"].to_first.map_idx):
                raise RuntimeError("The number of noised controls must be equal to the number of tau mapping.")
        self.n_collocation_points = n_collocation_points

        self.matrix_shape_k = (self.n_noised_controls, self.n_references)
        self.matrix_shape_c = (self.n_noised_states, self.n_noise)
        self.matrix_shape_a = (self.n_noised_states, self.n_noised_states)
        self.matrix_shape_cov = (self.n_noised_states, self.n_noised_states)
        self.matrix_shape_cov_cholesky = (self.n_noised_states, self.n_noised_states)
        self.matrix_shape_m = (self.n_noised_states, self.n_noised_states * self.n_collocation_points)
