
import biorbd_casadi as biorbd
import numpy as np

from bioptim.models.biorbd.external_forces import (
    ExternalForceSetTimeSeries,
    ExternalForceSetVariables,
)
from ...optimization.parameters import ParameterList
from .biorbd_model import BiorbdModel
from ..protocols.abstract_model_dynamics import TorqueDynamics


class TorqueBiorbdModel(BiorbdModel, TorqueDynamics):
    def __init__(
            self,
            bio_model: str | biorbd.Model,
            friction_coefficients: np.ndarray = None,
            parameters: ParameterList = None,
            external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None):
        BiorbdModel.__init__(self, bio_model, friction_coefficients, parameters, external_force_set)
        TorqueDynamics.__init__(self)

