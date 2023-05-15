import pytest
from bioptim import (
    BiorbdModel,
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
)
from .utils import TestUtils


def prepare_ocp(biorbd_model_path, phase_1, phase_2, assume_phase_dynamics) -> OptimalControlProgram:
    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path))

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
        MultinodeConstraintFcn.STATES_EQUALITY,
        nodes_phase=(phase_1, phase_2),
        nodes=(Node.START, Node.START),
    )
    multinode_constraints.add(
        MultinodeConstraintFcn.COM_EQUALITY,
        nodes_phase=(phase_1, phase_2),
        nodes=(Node.START, Node.START),
    )
    multinode_constraints.add(
        MultinodeConstraintFcn.COM_VELOCITY_EQUALITY,
        nodes_phase=(phase_1, phase_2),
        nodes=(Node.START, Node.START),
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))

    for bounds in x_bounds:
        for i in [1, 3, 4, 5]:
            bounds[i, [0, -1]] = 0
    x_bounds[0][2, 0] = 0.0
    x_bounds[2][2, [0, -1]] = [0.0, 1.57]

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model[0].nb_tau)
    u_init.add([tau_init] * bio_model[0].nb_tau)
    u_init.add([tau_init] * bio_model[0].nb_tau)

    return OptimalControlProgram(
        bio_model,
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
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("node", [*Node, 0])
def test_multinode_fail_first_node(node):
    # Constraints
    multinode_constraints = MultinodeConstraintList()
    # hard constraint
    if node in [Node.START, Node.MID, Node.PENULTIMATE, Node.END, 0]:
        multinode_constraints.add(
            MultinodeConstraintFcn.STATES_EQUALITY,
            nodes_phase=(0, 2),
            nodes=(node, Node.START),
        )
    else:
        with pytest.raises(
            NotImplementedError,
            match="Multinode Constraint only works with Node.START, Node.MID, Node.PENULTIMATE, Node.END or a int.",
        ):
            multinode_constraints.add(
                MultinodeConstraintFcn.STATES_EQUALITY,
                nodes_phase=(0, 2),
                nodes=(node, Node.START),
            )


@pytest.mark.parametrize("node", [*Node, 0])
def test_multinode_fail_second_node(node):
    # Constraints
    multinode_constraints = MultinodeConstraintList()
    # hard constraint
    if node in [Node.START, Node.MID, Node.PENULTIMATE, Node.END, 0]:
        multinode_constraints.add(
            MultinodeConstraintFcn.STATES_EQUALITY,
            nodes_phase=(0, 2),
            nodes=(node, Node.START),
        )
    else:
        with pytest.raises(
            NotImplementedError,
            match="Multinode Constraint only works with Node.START, Node.MID, Node.PENULTIMATE, Node.END or a int.",
        ):
            multinode_constraints.add(
                MultinodeConstraintFcn.STATES_EQUALITY,
                nodes_phase=(0, 2),
                nodes=(Node.START, node),
            )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("phase_1", [-1, 0, 4])
@pytest.mark.parametrize("phase_2", [-1, 0, 4])
def test_multinode_wrong_phase(phase_1, phase_2, assume_phase_dynamics):
    model = TestUtils.bioptim_folder() + "/examples/getting_started/models/cube.bioMod"

    # For reducing time assume_phase_dynamics=False is skipped for the failing tests

    if phase_1 == 4 or (phase_1 == 0 and phase_2 == 4) or (phase_1 == -1 and phase_2 == 4):
        with pytest.raises(
            RuntimeError,
            match="Phase index of the multinode_constraint is higher than the number of phases",
        ):
            prepare_ocp(model, phase_1, phase_2, assume_phase_dynamics=True)
    elif phase_1 == -1 or (phase_1 == 0 and phase_2 == -1):
        with pytest.raises(
            RuntimeError,
            match="Phase index of the multinode_constraint need to be positive",
        ):
            prepare_ocp(model, phase_1, phase_2, assume_phase_dynamics=True)
    else:
        prepare_ocp(model, phase_1, phase_2, assume_phase_dynamics=assume_phase_dynamics)
