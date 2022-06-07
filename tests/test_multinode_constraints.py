import pytest
from bioptim import (
    MultinodeConstraintList,
    MultinodeConstraintFcn,
    Node,
    OdeSolver,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    BoundsList,
    InitialGuessList,
    QAndQDotBounds,
)
import biorbd_casadi as biorbd
from .utils import TestUtils


def prepare_ocp(biorbd_model_path, phase_1, phase_2) -> OptimalControlProgram:
    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path))

    # Problem parameters
    n_shooting = (100, 300, 100)
    final_time = (2, 5, 4)
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    multinode_constraints = MultinodeConstraintList()
    # hard constraint
    multinode_constraints.add(
        MultinodeConstraintFcn.EQUALITY,
        phase_first_idx=phase_1,
        phase_second_idx=phase_2,
        first_node=Node.START,
        second_node=Node.START,
    )
    multinode_constraints.add(
        MultinodeConstraintFcn.COM_EQUALITY,
        phase_first_idx=phase_1,
        phase_second_idx=phase_2,
        first_node=Node.START,
        second_node=Node.START,
    )
    multinode_constraints.add(
        MultinodeConstraintFcn.COM_VELOCITY_EQUALITY,
        phase_first_idx=phase_1,
        phase_second_idx=phase_2,
        first_node=Node.START,
        second_node=Node.START,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    for bounds in x_bounds:
        for i in [1, 3, 4, 5]:
            bounds[i, [0, -1]] = 0
    x_bounds[0][2, 0] = 0.0
    x_bounds[2][2, [0, -1]] = [0.0, 1.57]

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        multinode_constraints=multinode_constraints,
        ode_solver=OdeSolver.RK4(),
    )


@pytest.mark.parametrize("node", [Node.ALL, Node.INTERMEDIATES, Node.ALL_SHOOTING])
def test_multinode_fail_first_node(node):
    # Constraints
    multinode_constraints = MultinodeConstraintList()
    # hard constraint
    with pytest.raises(
        NotImplementedError,
        match="Multi Node Constraint only works with Node.START, Node.MID, Node.PENULTIMATE, Node.END or a int.",
    ):
        multinode_constraints.add(
            MultinodeConstraintFcn.EQUALITY,
            phase_first_idx=0,
            phase_second_idx=2,
            first_node=node,
            second_node=Node.START,
        )


@pytest.mark.parametrize("node", [Node.ALL, Node.INTERMEDIATES, Node.ALL_SHOOTING])
def test_multinode_fail_second_node(node):
    # Constraints
    multinode_constraints = MultinodeConstraintList()
    # hard constraint
    with pytest.raises(
        NotImplementedError,
        match="Multi Node Constraint only works with Node.START, Node.MID, Node.PENULTIMATE, Node.END or a int.",
    ):
        multinode_constraints.add(
            MultinodeConstraintFcn.EQUALITY,
            phase_first_idx=0,
            phase_second_idx=2,
            first_node=Node.START,
            second_node=node,
        )


@pytest.mark.parametrize("phase_1", [-1, 0, 4])
@pytest.mark.parametrize("phase_2", [-1, 0, 4])
def test_multinode_wrong_phase(phase_1, phase_2):
    model = TestUtils.bioptim_folder() + "/examples/getting_started/models/cube.bioMod"
    if phase_1 == 4 or (phase_1 == 0 and phase_2 == 4) or (phase_1 == -1 and phase_2 == 4):
        with pytest.raises(
            RuntimeError,
            match="Phase index of the multinode_penalty is higher than the number of phases",
        ):
            prepare_ocp(model, phase_1, phase_2)
    elif phase_1 == -1 or (phase_1 == 0 and phase_2 == -1):
        with pytest.raises(
            RuntimeError,
            match="Phase index of the multinode_penalty need to be positive",
        ):
            prepare_ocp(model, phase_1, phase_2)
    else:
        prepare_ocp(model, phase_1, phase_2)
