from typing import Callable

import biorbd_casadi as biorbd
from casadi import MX, DM, inv
import numpy as np

from ..misc.utils import check_version
from ..misc.mapping import BiMappingList
from .biorbd_model import BiorbdModel

check_version(biorbd, "1.10.0", "1.11.0")


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

        self.n_references = n_references
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
