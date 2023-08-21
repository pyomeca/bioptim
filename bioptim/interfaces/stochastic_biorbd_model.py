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
    ):
        super().__init__(bio_model if isinstance(bio_model, str) else bio_model.model)

        self.motor_noise_magnitude = motor_noise_magnitude
        self.sensory_noise_magnitude = sensory_noise_magnitude
        self.sensory_reference_function = sensory_reference_function
        self.motor_noise_sym = MX.sym("motor_noise", motor_noise_magnitude.shape[0])
        self.sensory_noise_sym = MX.sym("sensory_noise", sensory_noise_magnitude.shape[0])
        self.motor_noise_mapping = motor_noise_mapping
