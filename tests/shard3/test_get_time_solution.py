import os

import pytest
import numpy as np
from casadi import Function, vertcat, DM

from bioptim import OdeSolver, Solver, PhaseDynamics, SolutionMerge, TimeAlignment, ControlType, Solution


def _get_solution(ode_solver, control_type, collocation_type, duplicate_first, is_multi_phase) -> Solution | None:
    if ode_solver == OdeSolver.COLLOCATION:
        if collocation_type == "none":
            return None  # This collocation needs a specific collocation type
        ode_solver_instance = ode_solver(method=collocation_type, duplicate_starting_point=duplicate_first)

    else:
        if collocation_type != "none" or duplicate_first:
            return None  # This test is only for collocation
        ode_solver_instance = ode_solver()

    # Load the appropriate ocp
    if is_multi_phase:
        from bioptim.examples.getting_started import example_multiphase as ocp_module

        bioptim_folder = os.path.dirname(ocp_module.__file__)
        model_path = bioptim_folder + "/models/cube.bioMod"
        prepare_args = {
            "biorbd_model_path": model_path,
            "ode_solver": ode_solver_instance,
            "control_type": control_type,
        }
    else:
        from bioptim.examples.getting_started import pendulum as ocp_module

        bioptim_folder = os.path.dirname(ocp_module.__file__)
        model_path = bioptim_folder + "/models/pendulum.bioMod"
        prepare_args = {
            "biorbd_model_path": model_path,
            "final_time": 2,
            "n_shooting": 10,
            "ode_solver": ode_solver_instance,
            "phase_dynamics": PhaseDynamics.SHARED_DURING_THE_PHASE,
            "control_type": control_type,
            "use_sx": False,
        }

    if ode_solver in (OdeSolver.COLLOCATION, OdeSolver.IRK) and control_type == ControlType.LINEAR_CONTINUOUS:
        with pytest.raises(
            NotImplementedError, match="ControlType.LINEAR_CONTINUOUS ControlType not implemented yet with COLLOCATION"
        ):
            ocp_module.prepare_ocp(**prepare_args)
        return None

    ocp = ocp_module.prepare_ocp(**prepare_args)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(0)
    solver.set_print_level(0)

    return ocp.solve(solver=solver)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
@pytest.mark.parametrize(
    "control_type", [ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE]
)
@pytest.mark.parametrize("collocation_type", ["none", "radau", "legendre"])
@pytest.mark.parametrize("duplicate_first", [False, True])
def test_get_time_aligned_with_states_single_phase(ode_solver, control_type, collocation_type, duplicate_first):
    sol = _get_solution(ode_solver, control_type, collocation_type, duplicate_first, is_multi_phase=False)
    if sol is None:
        return

    # Test all the merged combinations against the time for the decision variables
    states = sol.decision_states(to_merge=[])
    time = sol.decision_time(to_merge=[], time_alignment=TimeAlignment.STATES)
    assert len(time) == len(states["q"])
    for t, q in zip(time, states["q"]):
        assert t.shape[0] == q.shape[1]

    states = sol.decision_states(to_merge=SolutionMerge.KEYS)
    time = sol.decision_time(to_merge=SolutionMerge.KEYS, time_alignment=TimeAlignment.STATES)
    assert len(time) == len(states)
    for t, q in zip(time, states):
        assert t.shape[0] == q.shape[1]

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    time = sol.decision_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
    assert time.shape[0] == states["q"].shape[1]

    states = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS])
    time = sol.decision_time(to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS], time_alignment=TimeAlignment.STATES)
    assert time.shape[0] == states.shape[1]

    # Test all the merged combinations against the time for the stepwise variables
    states = sol.stepwise_states(to_merge=[])
    time = sol.stepwise_time(to_merge=[], time_alignment=TimeAlignment.STATES)
    assert len(time) == len(states["q"])
    for t, q in zip(time, states["q"]):
        assert t.shape[0] == q.shape[1]

    states = sol.stepwise_states(to_merge=SolutionMerge.KEYS)
    time = sol.stepwise_time(to_merge=SolutionMerge.KEYS, time_alignment=TimeAlignment.STATES)
    assert len(time) == len(states)
    for t, q in zip(time, states):
        assert t.shape[0] == q.shape[1]

    states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    time = sol.stepwise_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
    assert time.shape[0] == states["q"].shape[1]

    states = sol.stepwise_states(to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS])
    time = sol.stepwise_time(to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS], time_alignment=TimeAlignment.STATES)
    assert time.shape[0] == states.shape[1]


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
@pytest.mark.parametrize(
    "control_type", [ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE]
)
@pytest.mark.parametrize("collocation_type", ["none", "radau", "legendre"])
@pytest.mark.parametrize("duplicate_first", [False, True])
def test_get_time_aligned_with_controls_single_phase(ode_solver, control_type, collocation_type, duplicate_first):
    sol = _get_solution(ode_solver, control_type, collocation_type, duplicate_first, is_multi_phase=False)
    if sol is None:
        return

    # Test all the merged combinations against the time for the decision variables
    controls = sol.decision_controls(to_merge=[])
    time = sol.decision_time(to_merge=[], time_alignment=TimeAlignment.CONTROLS)
    assert len(time) == len(controls["tau"])
    for t, tau in zip(time, controls["tau"]):
        assert t.shape[0] == tau.shape[1]

    controls = sol.decision_controls(to_merge=SolutionMerge.KEYS)
    time = sol.decision_time(to_merge=SolutionMerge.KEYS, time_alignment=TimeAlignment.CONTROLS)
    assert len(time) == len(controls)
    for t, tau in zip(time, controls):
        assert t.shape[0] == tau.shape[1]

    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    time = sol.decision_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.CONTROLS)
    assert time.shape[0] == controls["tau"].shape[1]

    controls = sol.decision_controls(to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS])
    time = sol.decision_time(to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS], time_alignment=TimeAlignment.CONTROLS)
    assert time.shape[0] == controls.shape[1]

    # Test all the merged combinations against the time for the stepwise variables
    controls = sol.stepwise_controls(to_merge=[])
    time = sol.stepwise_time(to_merge=[], time_alignment=TimeAlignment.CONTROLS)
    assert len(time) == len(controls["tau"])
    for t, tau in zip(time, controls["tau"]):
        assert t.shape[0] == tau.shape[1]

    controls = sol.stepwise_controls(to_merge=SolutionMerge.KEYS)
    time = sol.stepwise_time(to_merge=SolutionMerge.KEYS, time_alignment=TimeAlignment.CONTROLS)
    assert len(time) == len(controls)
    for t, tau in zip(time, controls):
        assert t.shape[0] == tau.shape[1]

    controls = sol.stepwise_controls(to_merge=SolutionMerge.NODES)
    time = sol.stepwise_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.CONTROLS)
    assert time.shape[0] == controls["tau"].shape[1]

    controls = sol.stepwise_controls(to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS])
    time = sol.stepwise_time(to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS], time_alignment=TimeAlignment.CONTROLS)
    assert time.shape[0] == controls.shape[1]


@pytest.mark.parametrize(
    "ode_solver",
    [
        # OdeSolver.RK4,
        OdeSolver.COLLOCATION,
        # OdeSolver.IRK,
    ],
)
@pytest.mark.parametrize(
    "control_type", [ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE]
)
@pytest.mark.parametrize(
    "collocation_type",
    [
        # "none",
        "radau",
        "legendre",
    ],
)
@pytest.mark.parametrize("duplicate_first", [False, True])
@pytest.mark.parametrize("continuous", [False, True])
def test_get_time_aligned_with_states_multi_phases(
    ode_solver, control_type, collocation_type, duplicate_first, continuous
):
    sol = _get_solution(ode_solver, control_type, collocation_type, duplicate_first, is_multi_phase=True)
    if sol is None:
        return

    # Test all the merged combinations against the time for the decision variables
    states_phases = sol.decision_states(to_merge=[])
    times_phases = sol.decision_time(to_merge=[], time_alignment=TimeAlignment.STATES, continuous=continuous)
    for phase, (time, states) in enumerate(zip(times_phases, states_phases)):
        assert len(time) == len(states["q"])
        for t, q in zip(time, states["q"]):
            assert t.shape[0] == q.shape[1]

        if phase != 0:
            if continuous:
                assert times_phases[phase][0][0] == times_phases[phase - 1][-1]
            else:
                assert times_phases[phase][0][0] == 0

    states_phases = sol.decision_states(to_merge=SolutionMerge.KEYS)
    times_phases = sol.decision_time(
        to_merge=SolutionMerge.KEYS, time_alignment=TimeAlignment.STATES, continuous=continuous
    )
    for phase, (time, states) in enumerate(zip(times_phases, states_phases)):
        assert len(time) == len(states)
        for t, q in zip(time, states):
            assert t.shape[0] == q.shape[1]

        if phase != 0:
            if continuous:
                assert times_phases[phase][0][0] == times_phases[phase - 1][-1]
            else:
                assert times_phases[phase][0][0] == 0

    states_phases = sol.decision_states(to_merge=SolutionMerge.NODES)
    times_phases = sol.decision_time(
        to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES, continuous=continuous
    )
    for phase, (time, states) in enumerate(zip(times_phases, states_phases)):
        assert time.shape[0] == states["q"].shape[1]

        if phase != 0:
            if continuous:
                assert times_phases[phase][0] == times_phases[phase - 1][-1]
            else:
                assert times_phases[phase][0] == 0

    states_phases = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS])
    times_phases = sol.decision_time(
        to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS], time_alignment=TimeAlignment.STATES, continuous=continuous
    )
    for phase, (time, states) in enumerate(zip(times_phases, states_phases)):
        assert time.shape[0] == states.shape[1]

        if phase != 0:
            if continuous:
                assert times_phases[phase][0] == times_phases[phase - 1][-1]
            else:
                assert times_phases[phase][0] == 0

    states = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])
    time = sol.decision_time(
        to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES], time_alignment=TimeAlignment.STATES, continuous=continuous
    )
    assert time.shape[0] == states["q"].shape[1]
    if continuous:
        assert time[-1] == 8
    else:
        assert time[-1] == 4

    states = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES, SolutionMerge.KEYS])
    time = sol.decision_time(
        to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES, SolutionMerge.KEYS],
        time_alignment=TimeAlignment.STATES,
        continuous=continuous,
    )
    assert time.shape[0] == states.shape[1]
    if continuous:
        assert time[-1] == 8
    else:
        assert time[-1] == 4

    # Test all the merged combinations against the time for the stepwise variables
    states_phases = sol.stepwise_states(to_merge=[])
    times_phases = sol.stepwise_time(to_merge=[], time_alignment=TimeAlignment.STATES, continuous=continuous)
    for phase, (time, states) in enumerate(zip(times_phases, states_phases)):
        assert len(time) == len(states["q"])
        for t, q in zip(time, states["q"]):
            assert t.shape[0] == q.shape[1]

        if phase != 0:
            if continuous:
                assert times_phases[phase][0][0] == times_phases[phase - 1][-1]
            else:
                assert times_phases[phase][0][0] == 0

    states_phases = sol.stepwise_states(to_merge=SolutionMerge.KEYS)
    times_phases = sol.stepwise_time(
        to_merge=SolutionMerge.KEYS, time_alignment=TimeAlignment.STATES, continuous=continuous
    )
    for phase, (time, states) in enumerate(zip(times_phases, states_phases)):
        assert len(time) == len(states)
        for t, q in zip(time, states):
            assert t.shape[0] == q.shape[1]

        if phase != 0:
            if continuous:
                assert times_phases[phase][0][0] == times_phases[phase - 1][-1]
            else:
                assert times_phases[phase][0][0] == 0

    states_phases = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    times_phases = sol.stepwise_time(
        to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES, continuous=continuous
    )
    for phase, (time, states) in enumerate(zip(times_phases, states_phases)):
        assert time.shape[0] == states["q"].shape[1]

        if phase != 0:
            if continuous:
                assert times_phases[phase][0] == times_phases[phase - 1][-1]
            else:
                assert times_phases[phase][0] == 0

    states_phases = sol.stepwise_states(to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS])
    times_phases = sol.stepwise_time(
        to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS], time_alignment=TimeAlignment.STATES, continuous=continuous
    )
    for phase, (time, states) in enumerate(zip(times_phases, states_phases)):
        assert time.shape[0] == states.shape[1]

        if phase != 0:
            if continuous:
                assert times_phases[phase][0][0] == times_phases[phase - 1][-1]
            else:
                assert times_phases[phase][0][0] == 0

    states = sol.stepwise_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])
    time = sol.stepwise_time(
        to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES], time_alignment=TimeAlignment.STATES, continuous=continuous
    )
    assert time.shape[0] == states["q"].shape[1]
    if continuous:
        assert time[-1] == 8
    else:
        assert time[-1] == 4

    states = sol.stepwise_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES, SolutionMerge.KEYS])
    time = sol.stepwise_time(
        to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES, SolutionMerge.KEYS],
        time_alignment=TimeAlignment.STATES,
        continuous=continuous,
    )
    assert time.shape[0] == states.shape[1]
    if continuous:
        assert time[-1] == 8
    else:
        assert time[-1] == 4
