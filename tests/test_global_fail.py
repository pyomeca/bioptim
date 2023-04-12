import pytest

from casadi import MX
from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    ConstraintList,
    InitialGuess,
)

from .utils import TestUtils


def test_custom_constraint_mx_fail():
    def custom_mx_fail(pn):
        if pn.u_scaled is None:
            return None
        u = pn.nlp.controls
        return MX.zeros(u[0].shape), u[0].cx_start, MX.zeros(u[0].shape)    # TODO: [0] to [node_index]

    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/getting_started/models/cube.bioMod"

    constraints = ConstraintList()
    constraints.add(custom_mx_fail, node=Node.ALL)

    x_init = InitialGuess([0] * 6)
    u_init = InitialGuess([0] * 3)

    ocp = OptimalControlProgram(
        BiorbdModel(model_path),
        Dynamics(DynamicsFcn.TORQUE_DRIVEN),
        30,
        2,
        constraints=constraints,
        x_init=x_init,
        u_init=u_init,
    )

    with pytest.raises(RuntimeError, match="Ipopt doesn't support SX/MX types in constraints bounds"):
        ocp.solve()
