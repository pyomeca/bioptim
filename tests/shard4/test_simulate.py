from sys import platform

from bioptim import Shooting, OdeSolver, SolutionIntegrator, Solver, PhaseDynamics, SolutionMerge, ControlType
from bioptim.models.biorbd.viewer_utils import _check_models_comes_from_same_super_class
import numpy as np
import numpy.testing as npt
import os
import pytest
import warnings

from ..utils import TestUtils


@pytest.mark.parametrize("scaled", [True, False])
def test_merge_combinations(scaled):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod", final_time=2, n_shooting=10
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    solver.set_maximum_iterations(1)
    sol = ocp.solve(solver)

    # Merges that includes PHASES
    with pytest.raises(ValueError, match="Merging must at least contain SolutionMerge.KEYS or SolutionMerge.NODES"):
        sol.decision_states(to_merge=SolutionMerge.PHASES)
    with pytest.raises(ValueError, match="Merging must at least contain SolutionMerge.KEYS or SolutionMerge.NODES"):
        sol.decision_states(to_merge=[SolutionMerge.PHASES])
    with pytest.raises(ValueError, match="to_merge must contain SolutionMerge.NODES when merging phases"):
        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.KEYS])
    sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
    sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.KEYS, SolutionMerge.NODES])

    # Merges that includes KEYS
    sol.decision_states(to_merge=SolutionMerge.KEYS, scaled=scaled)
    sol.decision_states(to_merge=[SolutionMerge.KEYS], scaled=scaled)
    sol.decision_states(to_merge=[SolutionMerge.KEYS, SolutionMerge.NODES], scaled=scaled)
    with pytest.raises(ValueError, match="to_merge must contain SolutionMerge.NODES when merging phases"):
        sol.decision_states(to_merge=[SolutionMerge.KEYS, SolutionMerge.PHASES], scaled=scaled)
    sol.decision_states(to_merge=[SolutionMerge.KEYS, SolutionMerge.PHASES, SolutionMerge.NODES], scaled=scaled)

    # Merges that includes NODES
    sol.decision_states(to_merge=SolutionMerge.NODES, scaled=scaled)
    sol.decision_states(to_merge=[SolutionMerge.NODES], scaled=scaled)
    sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS], scaled=scaled)
    sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES], scaled=scaled)
    sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES, SolutionMerge.KEYS], scaled=scaled)

    # Merges that includes ALL
    sol.decision_states(to_merge=SolutionMerge.ALL)
    sol.decision_states(to_merge=[SolutionMerge.ALL])


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_merge_phases_one_phase(phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=10,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)

    states = sol.stepwise_states(to_merge=[SolutionMerge.NODES])
    sol_merged = sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])

    n_steps = ocp.nlp[0].n_states_stepwise_steps(0)
    for key in states:
        npt.assert_almost_equal(sol_merged[key], states[key][:, ::n_steps])


def test_merge_phases_multi_phase():
    # Load pendulum
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        expand_dynamics=True,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)

    states = sol.stepwise_states(to_merge=[SolutionMerge.NODES])
    sol_merged = sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])

    n_steps = ocp.nlp[0].n_states_stepwise_steps(0)
    for key in states[0]:
        expected = np.concatenate([s[key][:, ::n_steps] for s in states], axis=1)
        npt.assert_almost_equal(sol_merged[key], expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_interpolate(phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    n_shooting = 10

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=n_shooting,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)
    n_frames = 100
    sol_interp = sol.interpolate(n_frames)

    states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    sol_interp_list = sol.interpolate([n_frames])

    shapes = (2, 2)
    n_steps = ocp.nlp[0].n_states_stepwise_steps(0)
    for i, key in enumerate(states):
        npt.assert_almost_equal(sol_interp[key][:, [0, -1]], states[key][:, [0, -1]])
        npt.assert_almost_equal(sol_interp_list[key][:, [0, -1]], states[key][:, [0, -1]])
        assert sol_interp[key].shape == (shapes[i], n_frames)
        assert sol_interp_list[key].shape == (shapes[i], n_frames)
        assert states[key].shape == (shapes[i], (n_shooting * n_steps) + 1)

    with pytest.raises(
        ValueError,
        match="n_frames should either be an int to merge_phases phases or a "
        "list of int of the number of phases dimension",
    ):
        sol.interpolate([n_frames, n_frames])


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
def test_interpolate_multiphases(ode_solver):
    # Load pendulum
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        ode_solver=ode_solver(),
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        expand_dynamics=True,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)
    n_frames = 100
    n_shooting = [20, 30, 20]

    states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    sol_interp = sol.interpolate([n_frames, n_frames, n_frames])
    shapes = (3, 3)

    decimal = 2 if ode_solver == OdeSolver.COLLOCATION else 8
    n_steps = ocp.nlp[0].n_states_stepwise_steps(0)
    for i, key in enumerate(states[0]):
        npt.assert_almost_equal(sol_interp[i][key][:, [0, -1]], states[i][key][:, [0, -1]], decimal=decimal)
        assert sol_interp[i][key].shape == (shapes[i], n_frames)
        assert states[i][key].shape == (shapes[i], (n_shooting[i] * n_steps) + 1)

    with pytest.raises(
        ValueError,
        match="n_frames should either be an int to merge_phases phases or a "
        "list of int of the number of phases dimension",
    ):
        sol.interpolate([n_frames, n_frames])


def test_interpolate_multiphases_merge_phase():
    # Load pendulum
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        expand_dynamics=True,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)
    n_frames = 100
    n_shooting = [20, 30, 20]

    states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    sol_interp = sol.interpolate(n_frames)
    shapes = (3, 3)

    n_steps = ocp.nlp[0].n_states_stepwise_steps(0)
    for i, key in enumerate(states[0]):
        expected = np.array([states[0][key][:, 0], states[-1][key][:, -1]]).T
        npt.assert_almost_equal(sol_interp[key][:, [0, -1]], expected)

        assert sol_interp[key].shape == (shapes[i], n_frames)
        assert states[i][key].shape == (shapes[i], (n_shooting[i] * n_steps) + 1)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("integrator", [SolutionIntegrator.SCIPY_RK45, SolutionIntegrator.OCP])
def test_integrate(integrator, ode_solver, phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    n_shooting = 30 if integrator == SolutionIntegrator.SCIPY_RK45 else 10

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=0.9,
        n_shooting=n_shooting,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)

    opts = {"shooting_type": Shooting.MULTIPLE, "integrator": integrator}
    if ode_solver == OdeSolver.COLLOCATION and integrator == SolutionIntegrator.OCP:
        with pytest.raises(
            ValueError,
            match="When the ode_solver of the Optimal Control Problem is OdeSolver.COLLOCATION, "
            "we cannot use the SolutionIntegrator.OCP.\n"
            "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
            " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE",
        ):
            sol.integrate(**opts)
        return

    states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    sol_integrated = sol.integrate(**opts, to_merge=SolutionMerge.NODES)
    for key in sol_integrated.keys():
        assert np.shape(sol_integrated[key])[1] == np.shape(sol.stepwise_time(to_merge=SolutionMerge.NODES))[0]

    shapes = (2, 2)
    decimal = 5 if integrator != SolutionIntegrator.OCP else 8
    n_steps = ocp.nlp[0].n_states_stepwise_steps(0)

    for i, key in enumerate(states.keys()):
        npt.assert_almost_equal(sol_integrated[key][:, [0, -1]], states[key][:, [0, -1]], decimal=decimal)

        assert sol_integrated[key].shape == (shapes[i], n_shooting * n_steps + 1)
        assert states[key].shape == (shapes[i], n_shooting * n_steps + 1)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
def test_integrate_single_shoot(ode_solver, phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    n_shooting = 10

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=0.9,
        n_shooting=n_shooting,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)

    opts = {"integrator": SolutionIntegrator.OCP}
    if ode_solver == OdeSolver.COLLOCATION:
        with pytest.raises(
            ValueError,
            match="When the ode_solver of the Optimal Control Problem is OdeSolver.COLLOCATION, "
            "we cannot use the SolutionIntegrator.OCP.\n"
            "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
            " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE",
        ):
            sol.integrate(**opts)
        return

    states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    sol_integrated = sol.integrate(**opts, to_merge=SolutionMerge.NODES)
    time = sol.stepwise_time(to_merge=SolutionMerge.NODES)
    for key in sol_integrated.keys():
        assert np.shape(sol_integrated[key])[1] == np.shape(time)[0]

    shapes = (2, 2)
    decimal = 1
    n_steps = ocp.nlp[0].n_states_stepwise_steps(0)
    for i, key in enumerate(states):
        npt.assert_almost_equal(sol_integrated[key][:, [0, -1]], states[key][:, [0, -1]], decimal=decimal)

        assert sol_integrated[key].shape == (shapes[i], n_shooting * n_steps + 1)
        assert states[key].shape == (shapes[i], n_shooting * n_steps + 1)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
def test_integrate_single_shoot_use_scipy(ode_solver, phase_dynamics):
    if ode_solver == OdeSolver.COLLOCATION and platform != "linux-64":
        # For some reason, the test fails on Mac
        warnings.warn("Test test_integrate_single_shoot_use_scipy skiped on Mac")
        return

    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    n_shooting = 10

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=0.9,
        n_shooting=n_shooting,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)

    opts = {"integrator": SolutionIntegrator.SCIPY_RK45, "shooting_type": Shooting.SINGLE}

    states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    sol_integrated = sol.integrate(**opts, to_merge=SolutionMerge.NODES)
    time = sol.stepwise_time(to_merge=SolutionMerge.NODES)
    for key in sol_integrated.keys():
        assert np.shape(sol_integrated[key])[1] == np.shape(time)[0]

    decimal = 1
    if ode_solver == OdeSolver.RK4:
        npt.assert_almost_equal(
            sol_integrated["q"][:, [0, -1]],
            np.array([[0.0, -0.40229917], [0.0, 2.66577734]]),
            decimal=decimal,
        )
        npt.assert_almost_equal(
            sol_integrated["qdot"][:, [0, -1]],
            np.array([[0.0, 4.09704146], [0.0, 4.54449186]]),
            decimal=decimal,
        )

    else:
        npt.assert_almost_equal(
            sol_integrated["q"][:, [0, -1]],
            np.array([[0.0, -0.93010486], [0.0, 1.25096783]]),
            decimal=decimal,
        )
        npt.assert_almost_equal(
            sol_integrated["qdot"][:, [0, -1]],
            np.array([[0.0, -0.78079849], [0.0, 1.89447328]]),
            decimal=decimal,
        )

    shapes = (2, 2)
    n_steps = ocp.nlp[0].n_states_stepwise_steps(0)
    assert sol_integrated["q"].shape == (shapes[0], n_shooting * n_steps + 1)
    assert sol_integrated["qdot"].shape == (shapes[1], n_shooting * n_steps + 1)

    if ode_solver == OdeSolver.RK4:
        npt.assert_almost_equal(
            sol_integrated["q"][:, ::n_steps],
            np.array(
                [
                    [
                        0.0,
                        0.33771737,
                        0.60745128,
                        0.77322807,
                        0.87923355,
                        0.75783664,
                        -0.39855413,
                        -0.78071335,
                        -0.9923451,
                        -0.92719046,
                        -0.40229917,
                    ],
                    [
                        0.0,
                        -0.33826953,
                        -0.59909116,
                        -0.72747641,
                        -0.76068201,
                        -0.56369461,
                        0.62924769,
                        1.23356971,
                        1.64774156,
                        2.09574642,
                        2.66577734,
                    ],
                ],
            ),
            decimal=decimal,
        )
        npt.assert_almost_equal(
            sol_integrated["qdot"][:, ::n_steps],
            np.array(
                [
                    [
                        0.0,
                        4.56061105,
                        2.00396203,
                        1.71628908,
                        0.67171827,
                        -4.17420278,
                        -9.3109149,
                        -1.09241789,
                        -3.74378463,
                        6.01186572,
                        4.09704146,
                    ],
                    [
                        0.0,
                        -4.52749096,
                        -1.8038578,
                        -1.06710062,
                        0.30405407,
                        4.80782728,
                        10.24044964,
                        4.893414,
                        4.12673905,
                        6.83563286,
                        4.54449186,
                    ],
                ]
            ),
            decimal=decimal,
        )

    if ode_solver == OdeSolver.COLLOCATION:
        b = bool(1)
        for i, key in enumerate(states):
            b = b * states[key].shape == (shapes[i], n_shooting * 5 + 1)
        assert b


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("shooting", [Shooting.SINGLE, Shooting.MULTIPLE, Shooting.SINGLE_DISCONTINUOUS_PHASE])
@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("integrator", [SolutionIntegrator.OCP, SolutionIntegrator.SCIPY_RK45])
@pytest.mark.parametrize("control_type", [ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS])
def test_integrate_all_cases(shooting, merge, integrator, ode_solver, phase_dynamics, control_type):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    n_shooting = 10 if integrator == SolutionIntegrator.OCP else 30

    if ode_solver == OdeSolver.COLLOCATION and control_type == ControlType.LINEAR_CONTINUOUS:
        with pytest.raises(
            NotImplementedError,
            match="ControlType.LINEAR_CONTINUOUS ControlType not implemented yet with COLLOCATION",
        ):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
                final_time=1,
                n_shooting=n_shooting,
                ode_solver=ode_solver(),
                phase_dynamics=phase_dynamics,
                expand_dynamics=True,
                control_type=control_type,
            )
        return

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=n_shooting,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
        control_type=control_type,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)

    opts = {
        "shooting_type": shooting,
        "integrator": integrator,
        "to_merge": [SolutionMerge.NODES, SolutionMerge.PHASES if merge else None],
    }
    if ode_solver == OdeSolver.COLLOCATION and integrator == SolutionIntegrator.OCP:
        with pytest.raises(
            ValueError,
            match="When the ode_solver of the Optimal Control Problem is OdeSolver.COLLOCATION, "
            "we cannot use the SolutionIntegrator.OCP.\n"
            "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
            " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE",
        ):
            sol.integrate(**opts)
        return

    states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    sol_integrated = sol.integrate(**opts)
    for key in sol_integrated.keys():
        assert np.shape(sol_integrated[key])[1] == np.shape(sol.stepwise_time(to_merge=SolutionMerge.NODES))[0]

    shapes = (2, 2)
    decimal = 0 if integrator != SolutionIntegrator.OCP or ode_solver == OdeSolver.COLLOCATION else 8
    n_steps = ocp.nlp[0].n_states_stepwise_steps(0)
    npt.assert_almost_equal(sol_integrated["q"][:, [0, -1]], states["q"][:, [0, -1]], decimal=decimal)
    for i, key in enumerate(states.keys()):
        assert sol_integrated[key].shape == (shapes[i], n_shooting * n_steps + 1)
        assert states[key].shape == (shapes[i], n_shooting * n_steps + 1)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("shooting", [Shooting.SINGLE, Shooting.MULTIPLE, Shooting.SINGLE_DISCONTINUOUS_PHASE])
@pytest.mark.parametrize("integrator", [SolutionIntegrator.OCP, SolutionIntegrator.SCIPY_RK45])
@pytest.mark.parametrize("control_type", [ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS])
def test_integrate_multiphase(shooting, integrator, ode_solver, control_type):
    # Load pendulum
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        ode_solver=ode_solver(),
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        expand_dynamics=True,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)
    n_shooting = [20, 30, 20]

    opts = {"shooting_type": shooting, "integrator": integrator, "to_merge": SolutionMerge.NODES}

    if ode_solver == OdeSolver.COLLOCATION and integrator == SolutionIntegrator.OCP:
        with pytest.raises(
            ValueError,
            match="When the ode_solver of the Optimal Control Problem is OdeSolver.COLLOCATION, "
            "we cannot use the SolutionIntegrator.OCP.\n"
            "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
            " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE",
        ):
            sol.integrate(**opts)
        return

    states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    sol_integrated = sol.integrate(**opts)
    shapes = (3, 3)
    states_shape_sum = 0
    time_shape_sum = 0
    for i in range(len(sol_integrated)):
        for key in sol_integrated[i].keys():
            states_shape_sum += np.shape(sol_integrated[i][key])[1]
    time = sol.stepwise_time(to_merge=SolutionMerge.NODES)
    for t in time:
        time_shape_sum += t.shape[0] * 2  # For q and qdot
    assert states_shape_sum == time_shape_sum

    decimal = 1 if integrator != SolutionIntegrator.OCP else 8
    n_steps = ocp.nlp[0].n_states_stepwise_steps(0)
    for i in range(len(sol_integrated)):
        for k, key in enumerate(states[i]):
            if integrator == SolutionIntegrator.OCP or shooting == Shooting.MULTIPLE:
                npt.assert_almost_equal(sol_integrated[i][key][:, [0, -1]], states[i][key][:, [0, -1]], decimal=decimal)

            if ode_solver != OdeSolver.COLLOCATION and (
                integrator == SolutionIntegrator.OCP or shooting == Shooting.MULTIPLE
            ):
                npt.assert_almost_equal(sol_integrated[i][key], states[i][key])

            assert sol_integrated[i][key].shape == (shapes[k], n_shooting[i] * n_steps + 1)
            assert states[i][key].shape == (shapes[k], n_shooting[i] * n_steps + 1)

    sol_integrated, sol_time = sol.integrate(
        shooting_type=shooting,
        integrator=integrator,
        to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES],
        duplicated_times=False,
        return_time=True,
    )

    assert len(sol_time) == len(np.unique(sol_time))
    for i in range(len(sol_integrated)):
        for k, key in enumerate(states[i]):
            assert len(sol_integrated[key][k]) == len(sol_time)

    sol_integrated, sol_time = sol.integrate(
        shooting_type=shooting,
        integrator=integrator,
        to_merge=[SolutionMerge.NODES],
        duplicated_times=False,
        return_time=True,
    )
    for i in range(len(sol_time)):
        assert len(sol_time[i]) == len(np.unique(sol_time[i]))
    for i in range(len(sol_integrated)):
        for k, key in enumerate(states[i]):
            assert len(sol_integrated[i][key][k]) == len(sol_time[i])


def test_check_models_comes_from_same_super_class():
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        expand_dynamics=True,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    solver.set_print_level(0)
    sol = ocp.solve(solver)

    _check_models_comes_from_same_super_class(sol.ocp.nlp)

    class FakeModel:
        def __init__(self):
            self.nothing = 0

    sol.ocp.nlp[0].model = FakeModel()

    with pytest.raises(
        RuntimeError,
        match="The animation is only available for compatible models. "
        "Here, the model of phase 0 is of type FakeModel "
        "and the model of phase 0 is of type BiorbdModel and they don't share the same super class.",
    ):
        _check_models_comes_from_same_super_class(sol.ocp.nlp)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("shooting", [Shooting.SINGLE, Shooting.MULTIPLE, Shooting.SINGLE_DISCONTINUOUS_PHASE])
@pytest.mark.parametrize("integrator", [SolutionIntegrator.OCP, SolutionIntegrator.SCIPY_RK45])
def test_integrate_multiphase_merged(shooting, integrator, ode_solver):
    # Load pendulum
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        ode_solver=ode_solver(),
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        expand_dynamics=True,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)

    opts = {
        "shooting_type": shooting,
        "integrator": integrator,
        "to_merge": [SolutionMerge.NODES, SolutionMerge.PHASES],
    }

    if ode_solver == OdeSolver.COLLOCATION and integrator == SolutionIntegrator.OCP:
        with pytest.raises(
            ValueError,
            match="When the ode_solver of the Optimal Control Problem is OdeSolver.COLLOCATION, "
            "we cannot use the SolutionIntegrator.OCP.\n"
            "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
            " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE",
        ):
            sol.integrate(**opts)
        return

    n_shooting = [20, 30, 20]
    states = sol.stepwise_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])
    sol_integrated = sol.integrate(**opts)

    for key in sol_integrated.keys():
        assert (
            np.shape(sol_integrated[key])[1]
            == sol.stepwise_time(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]).shape[0]
        )

    shapes = (3, 3)
    decimal = 0 if integrator != SolutionIntegrator.OCP else 8
    n_steps = ocp.nlp[0].n_states_stepwise_steps(0)
    for k, key in enumerate(states):
        expected = np.array([states[key][:, 0], states[key][:, -1]]).T
        npt.assert_almost_equal(sol_integrated[key][:, [0, -1]], expected, decimal=decimal)

        assert sol_integrated[key].shape == (shapes[k], sum(n_shooting) * n_steps + 3)
        assert states[key].shape == (shapes[k], sum(n_shooting) * n_steps + 3)
