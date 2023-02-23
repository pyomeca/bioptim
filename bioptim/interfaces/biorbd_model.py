import biorbd_casadi as biorbd

from ..misc.utils import check_version
from .multi_biorbd_model import MultiBiorbdModel

check_version(biorbd, "1.9.9", "1.10.0")


class BiorbdModel(MultiBiorbdModel):
    def __init__(self, bio_model: str | biorbd.Model):
        super(BiorbdModel, self).__init__(bio_model)
