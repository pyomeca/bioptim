import os

import pytest
import numpy as np
from casadi import Function, vertcat, DM

from bioptim import OdeSolver, Solver, PhaseDynamics, SolutionMerge, TimeAlignment, ControlType, Solution


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
def test_node_time(ode_solver, phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=10,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(0)
    solver.set_print_level(0)

    sol = ocp.solve(solver=solver)
    all_node_time = np.array([ocp.node_time(0, i) for i in range(ocp.nlp[0].ns + 1)])

    computed_t = Function("time", [nlp.dt for nlp in ocp.nlp], [vertcat(all_node_time)])(sol.t_span[0][-1])
    time = sol.decision_time()
    expected_t = DM([0] + [t[-1] for t in time][:-1])
    np.testing.assert_almost_equal(np.array(computed_t), np.array(expected_t))


def _get_solution(ode_solver, control_type, collocation_type, duplicate_first) -> Solution | None:
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    if ode_solver == OdeSolver.COLLOCATION:
        if collocation_type == "none":
            return None  # This collocation needs a specific collocation type
        ode_solver_instance = ode_solver(method=collocation_type, duplicate_starting_point=duplicate_first)

    else:
        if collocation_type != "none" or duplicate_first:
            return None  # This test is only for collocation
        ode_solver_instance = ode_solver()

    if ode_solver in (OdeSolver.COLLOCATION, OdeSolver.IRK) and control_type == ControlType.LINEAR_CONTINUOUS:
        with pytest.raises(NotImplementedError, match="ControlType.LINEAR_CONTINUOUS ControlType not implemented yet with COLLOCATION"):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
                final_time=2,
                n_shooting=10,
                ode_solver=ode_solver_instance,
                phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
                control_type=control_type,
                use_sx=False,
            )
        return None

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=10,
        ode_solver=ode_solver_instance,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        control_type=control_type,
        use_sx=False,
    )
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(0)
    solver.set_print_level(0)

    return ocp.solve(solver=solver)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
@pytest.mark.parametrize(
    "control_type",
    [ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE]
)
@pytest.mark.parametrize("collocation_type", ["none", "radau", "legendre"])
@pytest.mark.parametrize("duplicate_first", [False, True])
def test_get_time_aligned_with_states_single_phase(ode_solver, control_type, collocation_type, duplicate_first):
    sol = _get_solution(ode_solver, control_type, collocation_type, duplicate_first)
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
@pytest.mark.parametrize("control_type", [ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE])
@pytest.mark.parametrize("collocation_type", ["none", "radau", "legendre"])
@pytest.mark.parametrize("duplicate_first", [False, True])
def test_get_time_aligned_with_controls_single_phase(ode_solver, control_type, collocation_type, duplicate_first):
    sol = _get_solution(ode_solver, control_type, collocation_type, duplicate_first)
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

# TODO: Multiphase