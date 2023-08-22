from typing import Callable

import biorbd_casadi as biorbd
from casadi import MX, DM, inv
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
        n_noised_states: int,
        n_noised_controls: int,
        sensory_noise_magnitude: np.ndarray | DM,
        motor_noise_magnitude: np.ndarray | DM,
        sensory_reference: Callable,
        motor_noise_mapping: BiMappingList = BiMappingList(),
        n_collocation_points: int = 1,
        **kwargs,
    ):
        super().__init__(bio_model if isinstance(bio_model, str) else bio_model.model, **kwargs)

        self.motor_noise_magnitude = motor_noise_magnitude
        self.sensory_noise_magnitude = sensory_noise_magnitude

        self.sensory_reference = sensory_reference

        self.motor_noise_sym = MX.sym("motor_noise", motor_noise_magnitude.shape[0])
        self.sensory_noise_sym = MX.sym("sensory_noise", sensory_noise_magnitude.shape[0])
        self.motor_noise_mapping = motor_noise_mapping

        n_noise = motor_noise_magnitude.shape[0] + sensory_noise_magnitude.shape[0]
        if motor_noise_mapping is not None and "tau" in motor_noise_mapping:
            n_noised_controls = len(motor_noise_mapping["tau"].to_second.map_idx)

        self.matrix_shape_k = (n_noised_controls, n_references)
        self.matrix_shape_c = (n_noised_states, n_noise)
        self.matrix_shape_a = (n_noised_states, n_noised_states)
        self.matrix_shape_cov = (n_noised_states, n_noised_states)
        self.matrix_shape_cov_cholesky = (n_noised_states, n_noised_states)
        self.matrix_shape_m = (n_noised_states, n_noised_states * n_collocation_points)
