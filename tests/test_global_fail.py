import pytest

import biorbd
from casadi import vertcat, MX
from bioptim import (
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
        return MX.zeros(pn.u.shape), pn.u, MX.zeros(pn.u.shape)

    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/getting_started/cube.bioMod"
    constraints = ConstraintList()
    constraints.add(custom_mx_fail, node=Node.ALL)

    ocp = OptimalControlProgram(
        biorbd.Model(model_path),
        Dynamics(DynamicsFcn.TORQUE_DRIVEN),
        30,
        2,
        constraints=constraints,
    )

    with pytest.raises(RuntimeError, match="Ipopt doesn't support SX/MX types in constraints bounds"):
        sol = ocp.solve(show_online_optim=True)
