import re

import pytest
from bioptim import (
    TorqueBiorbdModel,
    DynamicsOptions,
    MultinodeConstraintList,
    MultinodeConstraintFcn,
    Node,
    OdeSolver,
    OptimalControlProgram,
    DynamicsOptionsList,
    ObjectiveList,
    BoundsList,
    PhaseDynamics,
)
from tests.utils import TestUtils


def prepare_ocp(biorbd_model_path, phase_1, phase_2, phase_dynamics) -> OptimalControlProgram:
    bio_model = (
        TorqueBiorbdModel(biorbd_model_path),
        TorqueBiorbdModel(biorbd_model_path),
        TorqueBiorbdModel(biorbd_model_path),
    )

    # Problem parameters
    n_shooting = (100, 300, 100)
    final_time = (2, 5, 4)
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.RK4(), expand_dynamics=True, phase_dynamics=phase_dynamics))
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.RK4(), expand_dynamics=True, phase_dynamics=phase_dynamics))
    dynamics.add(DynamicsOptions(ode_solver=OdeSolver.RK4(), expand_dynamics=True, phase_dynamics=phase_dynamics))

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
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)

    for bounds in x_bounds:
        bounds["q"][1, [0, -1]] = 0
        bounds["qdot"][:, [0, -1]] = 0
    x_bounds[0]["q"][2, 0] = 0.0
    x_bounds[2]["q"][2, [0, -1]] = [0.0, 1.57]

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=0)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[1].nb_tau, max_bound=[tau_max] * bio_model[1].nb_tau, phase=1)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[2].nb_tau, max_bound=[tau_max] * bio_model[2].nb_tau, phase=2)

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        multinode_constraints=multinode_constraints,
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
            ValueError,
            match=re.escape(
                "Multinode penalties only works with Node.START, Node.MID, Node.PENULTIMATE, Node.END or a node index (int)."
            ),
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
            ValueError,
            match=re.escape(
                "Multinode penalties only works with Node.START, Node.MID, Node.PENULTIMATE, Node.END or a node index (int)."
            ),
        ):
            multinode_constraints.add(
                MultinodeConstraintFcn.STATES_EQUALITY,
                nodes_phase=(0, 2),
                nodes=(Node.START, node),
            )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE])
@pytest.mark.parametrize("phase_1", [-1, 0, 4])
@pytest.mark.parametrize("phase_2", [-1, 1, 4])
def test_multinode_wrong_phase(phase_1, phase_2, phase_dynamics):
    model = TestUtils.bioptim_folder() + "/examples/models/cube.bioMod"

    if phase_1 == 4 or (phase_1 == 0 and phase_2 == 4) or (phase_1 == -1 and phase_2 == 4):
        with pytest.raises(
            ValueError,
            match="nodes_phase of the multinode_penalty must be between 0 and number of phases",
        ):
            prepare_ocp(model, phase_1, phase_2, phase_dynamics=phase_dynamics)
    elif phase_1 == -1 or (phase_1 == 0 and phase_2 == -1):
        with pytest.raises(
            ValueError,
            match="nodes_phase of the multinode_penalty must be between 0 and number of phases",
        ):
            prepare_ocp(model, phase_1, phase_2, phase_dynamics=phase_dynamics)
    else:
        prepare_ocp(model, phase_1, phase_2, phase_dynamics=phase_dynamics)
