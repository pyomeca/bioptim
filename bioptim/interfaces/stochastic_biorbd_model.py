import biorbd_casadi as biorbd
from casadi import MX, DM
import numpy as np

from ..misc.utils import check_version
from ..misc.mapping import BiMappingList
from .biorbd_model import BiorbdModel

check_version(biorbd, "1.9.9", "1.10.0")


class StochasticBiorbdModel(BiorbdModel):
    """
    This class allows to define a biorbd model.
    """

    def __init__(
        self,
        bio_model: str | BiorbdModel,
        sensory_noise_magnitude: np.ndarray | DM,
        motor_noise_magnitude: np.ndarray | DM,
        sensory_reference_function: callable,
        motor_noise_mapping: BiMappingList = BiMappingList(),
        n_references: int = 0,
        n_collocation_points: int = 1,
    ):
        super().__init__(bio_model if isinstance(bio_model, str) else bio_model.model)

        self.motor_noise_magnitude = motor_noise_magnitude
        self.sensory_noise_magnitude = sensory_noise_magnitude
        self.sensory_reference_function = sensory_reference_function
        self.motor_noise_sym = MX.sym("motor_noise", motor_noise_magnitude.shape[0])
        self.sensory_noise_sym = MX.sym("sensory_noise", sensory_noise_magnitude.shape[0])
        self.motor_noise_mapping = motor_noise_mapping

        self.friction = np.array([[0.05, 0.025], [0.025, 0.05]])

        # TODO: this should be changed when other dynamics are implemented
        n_noised_states = bio_model.nb_q - bio_model.nb_root
        n_noise = motor_noise_magnitude.shape[0] + sensory_noise_magnitude.shape[0]

        n_noised_controls = bio_model.nb_tau
        if motor_noise_mapping is not None and "tau" in motor_noise_mapping:
            n_noised_controls = len(motor_noise_mapping["tau"].to_second.map_idx)
        self.matrix_shape_k = (n_noised_controls, n_references)
        self.matrix_shape_c = (n_noised_states, n_noise)
        self.matrix_shape_a = (n_noised_states, n_noised_states)
        self.matrix_shape_cov = (n_noised_states, n_noised_states)
        self.matrix_shape_cov_cholesky = (n_noised_states, n_noised_states)
        self.matrix_shape_m = (n_noised_states, n_noised_states * n_collocation_points)

