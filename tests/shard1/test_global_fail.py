import pytest
import re

from casadi import MX
from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    ConstraintList,
    InitialGuessList,
    PenaltyController,
    PhaseDynamics,
)

from tests.utils import TestUtils


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

    x_init = InitialGuessList()
    x_init["q"] = [0] * 3
    x_init["qdot"] = [0] * 3
    u_init = InitialGuessList()
    u_init["tau"] = [0] * 3

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "You cannot have non linear bounds for custom constraints and min_bound or max_bound defined.\n"
            "Please note that you may run into this error message if phase_dynamics "
            "was set to PhaseDynamics.ONE_PER_NODE. One workaround is to define your penalty one node at a "
            "time instead of using the built-in ALL_SHOOTING (or something similar)."
        ),
    ):
        OptimalControlProgram(
            BiorbdModel(model_path),
            Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, phase_dynamics=PhaseDynamics.ONE_PER_NODE),
            30,
            2,
            constraints=constraints,
            x_init=x_init,
            u_init=u_init,
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_custom_constraint_mx_fail(phase_dynamics):
    def custom_mx_fail(controller: PenaltyController):
        if controller.u_scaled is None:
            return None
        u = controller.controls
        return MX.zeros(u.shape), u.cx_start, MX.zeros(u.shape)

    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/getting_started/models/cube.bioMod"

    constraints = ConstraintList()
    constraints.add(custom_mx_fail, node=0)

    x_init = InitialGuessList()
    x_init["q"] = [0] * 3
    x_init["qdot"] = [0] * 3
    u_init = InitialGuessList()
    u_init["tau"] = [0] * 3

    ocp = OptimalControlProgram(
        BiorbdModel(model_path),
        Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, phase_dynamics=phase_dynamics),
        30,
        2,
        constraints=constraints,
        x_init=x_init,
        u_init=u_init,
    )

    with pytest.raises(RuntimeError, match="Ipopt doesn't support SX/MX types in constraints bounds"):
        ocp.solve()
