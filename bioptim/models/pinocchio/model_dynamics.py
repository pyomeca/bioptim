from .pinocchio_model import PinocchioModel
from ...dynamics.state_space_dynamics import TorqueDynamics


class TorquePinocchioModel(PinocchioModel, TorqueDynamics):
    def __init__(self, bio_model: str | object, **kwargs):
        super().__init__(bio_model=bio_model, **kwargs)
