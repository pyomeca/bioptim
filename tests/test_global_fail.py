import pytest

from casadi import MX
from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    ConstraintList,
)

from .utils import TestUtils


def test_custom_constraint_mx_fail():
    def custom_mx_fail(pn):
        if pn.u is None:
            return None
        u = pn.nlp.controls
        return MX.zeros(u.shape), u.cx, MX.zeros(u.shape)

    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/getting_started/models/cube.bioMod"

    constraints = ConstraintList()
    constraints.add(custom_mx_fail, node=Node.ALL)

    ocp = OptimalControlProgram(
        BiorbdModel(model_path), Dynamics(DynamicsFcn.TORQUE_DRIVEN), 30, 2, constraints=constraints
    )

    with pytest.raises(RuntimeError, match="Ipopt doesn't support SX/MX types in constraints bounds"):
        ocp.solve()
