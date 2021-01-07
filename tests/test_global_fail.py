import pytest

from pathlib import Path

import biorbd
from casadi import vertcat, MX

from bioptim import (
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    ConstraintList,
)


def test_custom_constraint_mx_fail():
    def custom_mx_fail(ocp, nlp, t, x, u, p):
        return MX(0), vertcat(*u), MX(0)

    PROJECT_FOLDER = Path(__file__).parent / ".."
    model_path = str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod"
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
