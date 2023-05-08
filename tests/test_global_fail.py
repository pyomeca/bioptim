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
    PenaltyController
)

from .utils import TestUtils


def test_custom_constraint_multiple_nodes_fail():
    def custom_mx_fail(controller: PenaltyController):
        if controller.u_scaled is None:
            return None
        u = controller.controls
        return MX.zeros(u.shape), u.cx_start, MX.zeros(u.shape)

    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/getting_started/models/cube.bioMod"

    constraints = ConstraintList()
    constraints.add(custom_mx_fail, node=Node.ALL)

    x_init = InitialGuess([0] * 6)
    u_init = InitialGuess([0] * 3)

    with pytest.raises(
            NotImplementedError,
            match="Setting custom function for more than one node at a time when assume_phase_dynamics "
                  "is set to False is not Implemented"
    ):
        OptimalControlProgram(
            BiorbdModel(model_path),
            Dynamics(DynamicsFcn.TORQUE_DRIVEN),
            30,
            2,
            constraints=constraints,
            x_init=x_init,
            u_init=u_init,
        )


def test_custom_constraint_mx_fail():
    def custom_mx_fail(controller: PenaltyController):
        if controller.u_scaled is None:
            return None
        u = controller.controls
        return MX.zeros(u.shape), u.cx_start, MX.zeros(u.shape)

    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/getting_started/models/cube.bioMod"

    constraints = ConstraintList()
    constraints.add(custom_mx_fail, node=0)

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
