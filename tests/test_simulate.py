import warnings
import os
from sys import platform
import pytest

import numpy as np
from bioptim import Shooting, OdeSolver, SolutionIntegrator, Solver


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_merge_phases_one_phase(assume_phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=10,
        assume_phase_dynamics=assume_phase_dynamics,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)
    sol_merged = sol.merge_phases()
    for key in sol.states:
        np.testing.assert_almost_equal(sol_merged.states[key], sol.states[key])
    for key in sol.controls:
        np.testing.assert_almost_equal(sol_merged.controls[key], sol.controls[key])


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_merge_phases_multi_phase(assume_phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        assume_phase_dynamics=assume_phase_dynamics,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)
    sol_merged = sol.merge_phases()

    for key in sol.states[0]:
        expected = np.concatenate([s[key][:, :-1] for s in sol.states], axis=1)
        expected = np.concatenate((expected, sol.states[-1][key][:, -1][:, np.newaxis]), axis=1)

        np.testing.assert_almost_equal(sol_merged.states[key], expected)

    for key in sol.controls[0]:
        expected = np.concatenate([s[key][:, :-1] for s in sol.controls], axis=1)
        expected = np.concatenate((expected, sol.controls[-1][key][:, -1][:, np.newaxis]), axis=1)

        np.testing.assert_almost_equal(sol_merged.controls[key], expected)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_interpolate(assume_phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    n_shooting = 10

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=n_shooting,
        assume_phase_dynamics=assume_phase_dynamics,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)
    n_frames = 100
    sol_interp = sol.interpolate(n_frames)
    sol_interp_list = sol.interpolate([n_frames])
    shapes = (2, 2)
    for i, key in enumerate(sol.states):
        np.testing.assert_almost_equal(sol_interp.states[key][:, [0, -1]], sol.states[key][:, [0, -1]])
        np.testing.assert_almost_equal(sol_interp_list.states[key][:, [0, -1]], sol.states[key][:, [0, -1]])
        assert sol_interp.states[key].shape == (shapes[i], n_frames)
        assert sol_interp_list.states[key].shape == (shapes[i], n_frames)
        assert sol.states[key].shape == (shapes[i], n_shooting + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_interp.controls

    with pytest.raises(
        ValueError,
        match="n_frames should either be a int to merge_phases phases or a "
        "list of int of the number of phases dimension",
    ):
        sol.interpolate([n_frames, n_frames])


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
def test_interpolate_multiphases(ode_solver, assume_phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        ode_solver=ode_solver(),
        assume_phase_dynamics=assume_phase_dynamics,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)
    n_frames = 100
    n_shooting = [20, 30, 20]
    sol_interp = sol.interpolate([n_frames, n_frames, n_frames])
    shapes = (3, 3)

    decimal = 2 if ode_solver == OdeSolver.COLLOCATION else 8
    for i, key in enumerate(sol.states[0]):
        np.testing.assert_almost_equal(
            sol_interp.states[i][key][:, [0, -1]], sol.states[i][key][:, [0, -1]], decimal=decimal
        )
        assert sol_interp.states[i][key].shape == (shapes[i], n_frames)
        if ode_solver == OdeSolver.COLLOCATION:
            assert sol.states[i][key].shape == (shapes[i], n_shooting[i] * 5 + 1)
        else:
            assert sol.states[i][key].shape == (shapes[i], n_shooting[i] + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_interp.controls

    with pytest.raises(
        ValueError,
        match="n_frames should either be a int to merge_phases phases or a "
        "list of int of the number of phases dimension",
    ):
        sol.interpolate([n_frames, n_frames])


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_interpolate_multiphases_merge_phase(assume_phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        assume_phase_dynamics=assume_phase_dynamics,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)
    n_frames = 100
    n_shooting = [20, 30, 20]
    sol_interp = sol.interpolate(n_frames)
    shapes = (3, 3)

    for i, key in enumerate(sol.states[0]):
        expected = np.array([sol.states[0][key][:, 0], sol.states[-1][key][:, -1]]).T
        np.testing.assert_almost_equal(sol_interp.states[key][:, [0, -1]], expected)

        assert sol_interp.states[key].shape == (shapes[i], n_frames)
        assert sol.states[i][key].shape == (shapes[i], n_shooting[i] + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_interp.controls


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("integrator", [SolutionIntegrator.SCIPY_RK45, SolutionIntegrator.OCP])
def test_integrate(integrator, ode_solver, assume_phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    n_shooting = 30 if integrator == SolutionIntegrator.SCIPY_RK45 else 10

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=0.9,
        n_shooting=n_shooting,
        ode_solver=ode_solver(),
        assume_phase_dynamics=assume_phase_dynamics,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)

    opts = {"shooting_type": Shooting.MULTIPLE, "keep_intermediate_points": False, "integrator": integrator}
    with pytest.raises(
        ValueError,
        match="shooting_type=Shooting.MULTIPLE and keep_intermediate_points=False cannot be used simultaneously."
        "When using multiple shooting, the intermediate points should be kept.",
    ):
        _ = sol.integrate(**opts)

    opts["keep_intermediate_points"] = True
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

    sol_integrated = sol.integrate(**opts)
    for key in sol_integrated.states.keys():
        assert np.shape(sol_integrated.states[key])[1] == np.shape(sol_integrated.time)[0]

    shapes = (2, 2)
    decimal = 5 if integrator != SolutionIntegrator.OCP else 8
    for i, key in enumerate(sol.states):
        np.testing.assert_almost_equal(
            sol_integrated.states[key][:, [0, -1]], sol.states[key][:, [0, -1]], decimal=decimal
        )

        assert sol_integrated.states[key].shape == (shapes[i], n_shooting * 6 + 1)
        if ode_solver == OdeSolver.COLLOCATION:
            assert sol.states[key].shape == (shapes[i], n_shooting * 5 + 1)
        else:
            assert sol.states[key].shape == (shapes[i], n_shooting + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. "
        "This may happen in previously integrated and interpolated structure",
    ):
        _ = sol_integrated.controls


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("keep_intermediate_points", [False, True])
def test_integrate_single_shoot(keep_intermediate_points, ode_solver, assume_phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    n_shooting = 10

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=0.9,
        n_shooting=n_shooting,
        ode_solver=ode_solver(),
        assume_phase_dynamics=assume_phase_dynamics,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)

    opts = {"keep_intermediate_points": keep_intermediate_points, "integrator": SolutionIntegrator.OCP}
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

    sol_integrated = sol.integrate(**opts)
    for key in sol_integrated.states.keys():
        assert np.shape(sol_integrated.states[key])[1] == np.shape(sol_integrated._time_vector)[1]

    shapes = (2, 2)
    decimal = 1
    for i, key in enumerate(sol.states):
        np.testing.assert_almost_equal(
            sol_integrated.states[key][:, [0, -1]], sol.states[key][:, [0, -1]], decimal=decimal
        )

        if keep_intermediate_points or ode_solver == OdeSolver.COLLOCATION:
            assert sol_integrated.states[key].shape == (shapes[i], n_shooting * 5 + 1)
        else:
            np.testing.assert_almost_equal(sol_integrated.states[key], sol.states[key], decimal=decimal)
            assert sol_integrated.states[key].shape == (shapes[i], n_shooting + 1)

        assert sol.states[key].shape == (shapes[i], n_shooting + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_integrated.controls


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("keep_intermediate_points", [False, True])
def test_integrate_single_shoot_use_scipy(keep_intermediate_points, ode_solver, assume_phase_dynamics):
    if ode_solver == OdeSolver.COLLOCATION and platform != "linux-64":
        # For some reason, the test fails on Mac
        warnings.warn("Test test_integrate_single_shoot_use_scipy skiped on Mac")
        return

    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    n_shooting = 10

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=0.9,
        n_shooting=n_shooting,
        ode_solver=ode_solver(),
        assume_phase_dynamics=assume_phase_dynamics,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)

    opts = {
        "keep_intermediate_points": keep_intermediate_points,
        "integrator": SolutionIntegrator.SCIPY_RK45,
        "shooting_type": Shooting.SINGLE,
    }

    sol_integrated = sol.integrate(**opts)
    assert np.shape(sol_integrated.states["all"])[1] == np.shape(sol_integrated.time)[0]

    decimal = 1
    if ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(
            sol_integrated.states["q"][:, [0, -1]],
            np.array([[0.0, -0.40229917], [0.0, 2.66577734]]),
            decimal=decimal,
        )
        np.testing.assert_almost_equal(
            sol_integrated.states["qdot"][:, [0, -1]],
            np.array([[0.0, 4.09704146], [0.0, 4.54449186]]),
            decimal=decimal,
        )

    else:
        np.testing.assert_almost_equal(
            sol_integrated.states["q"][:, [0, -1]],
            np.array([[0.0, -0.93010486], [0.0, 1.25096783]]),
            decimal=decimal,
        )
        np.testing.assert_almost_equal(
            sol_integrated.states["qdot"][:, [0, -1]],
            np.array([[0.0, -0.78079849], [0.0, 1.89447328]]),
            decimal=decimal,
        )

    shapes = (2, 2)
    if keep_intermediate_points:
        assert sol_integrated.states["q"].shape == (shapes[1], n_shooting * 5 + 1)
        assert sol_integrated.states["qdot"].shape == (shapes[2], n_shooting * 5 + 1)
    else:
        if ode_solver == OdeSolver.RK4:
            np.testing.assert_almost_equal(
                sol_integrated.states["all"],
                np.array(
                    [
                        [0.0, 0.3, 0.6, 0.8, 0.9, 0.8, -0.4, -0.8, -1.0, -0.9, -0.4],
                        [0.0, -0.3, -0.6, -0.7, -0.8, -0.6, 0.6, 1.2, 1.6, 2.1, 2.7],
                        [0.0, 4.6, 2.0, 1.7, 0.7, -4.2, -9.3, -1.1, -3.7, 6.0, 4.1],
                        [0.0, -4.5, -1.8, -1.1, 0.3, 4.8, 10.2, 4.9, 4.1, 6.8, 4.5],
                    ]
                ),
                decimal=decimal,
            )
            np.testing.assert_almost_equal(
                sol_integrated.states["q"],
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
            np.testing.assert_almost_equal(
                sol_integrated.states["qdot"],
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
        assert (
            sol_integrated.states["all"].shape == (shapes[0], n_shooting + 1)
            and sol_integrated.states["q"].shape == (shapes[1], n_shooting + 1)
            and sol_integrated.states["qdot"].shape == (shapes[2], n_shooting + 1)
        )

    if ode_solver == OdeSolver.COLLOCATION:
        b = bool(1)
        for i, key in enumerate(sol.states):
            b = b * sol.states[key].shape == (shapes[i], n_shooting * 5 + 1)
        assert b

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_integrated.controls


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("shooting", [Shooting.SINGLE, Shooting.MULTIPLE, Shooting.SINGLE_DISCONTINUOUS_PHASE])
@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("integrator", [SolutionIntegrator.OCP, SolutionIntegrator.SCIPY_RK45])
def test_integrate_all_cases(shooting, merge, integrator, ode_solver, assume_phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    n_shooting = 10 if integrator == SolutionIntegrator.OCP else 30

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=n_shooting,
        ode_solver=ode_solver(),
        assume_phase_dynamics=assume_phase_dynamics,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)

    opts = {
        "shooting_type": shooting,
        "keep_intermediate_points": False,
        "integrator": integrator,
    }

    if shooting == Shooting.MULTIPLE:
        with pytest.raises(
            ValueError,
            match="shooting_type=Shooting.MULTIPLE and keep_intermediate_points=False cannot be used simultaneously."
            "When using multiple shooting, the intermediate points should be kept.",
        ):
            _ = sol.integrate(**opts)

    opts["keep_intermediate_points"] = True
    opts["merge_phases"] = merge
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

    sol_integrated = sol.integrate(**opts)
    for key in sol_integrated.states.keys():
        assert np.shape(sol_integrated.states[key])[1] == np.shape(sol_integrated._time_vector)[1]

    shapes = (2, 2)
    decimal = 0 if integrator != SolutionIntegrator.OCP or ode_solver == OdeSolver.COLLOCATION else 8
    np.testing.assert_almost_equal(sol_integrated.states["q"][:, [0, -1]], sol.states["q"][:, [0, -1]], decimal=decimal)
    for i, key in enumerate(sol.states):
        if ode_solver == OdeSolver.COLLOCATION:
            if integrator != SolutionIntegrator.OCP:
                if shooting == Shooting.MULTIPLE:
                    assert sol_integrated.states[key].shape == (shapes[i], n_shooting * 6 + 1)
                else:
                    assert sol_integrated.states[key].shape == (shapes[i], n_shooting * 5 + 1)
            else:
                assert sol_integrated.states[key].shape == (shapes[i], n_shooting * (4 + 1) + 1)
            assert sol.states[key].shape == (shapes[i], n_shooting * 5 + 1)
        else:
            if shooting == Shooting.MULTIPLE:
                assert sol_integrated.states[key].shape == (shapes[i], n_shooting * (5 + 1) + 1)
            else:
                assert sol_integrated.states[key].shape == (shapes[i], n_shooting * 5 + 1)
            assert sol.states[key].shape == (shapes[i], n_shooting + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_integrated.controls


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("shooting", [Shooting.SINGLE, Shooting.MULTIPLE, Shooting.SINGLE_DISCONTINUOUS_PHASE])
@pytest.mark.parametrize("keep_intermediate_points", [True, False])
@pytest.mark.parametrize("integrator", [SolutionIntegrator.OCP, SolutionIntegrator.SCIPY_RK45])
def test_integrate_multiphase(shooting, keep_intermediate_points, integrator, ode_solver, assume_phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        ode_solver=ode_solver(),
        assume_phase_dynamics=assume_phase_dynamics,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)
    n_shooting = [20, 30, 20]

    opts = {
        "shooting_type": shooting,
        "keep_intermediate_points": keep_intermediate_points,
        "integrator": integrator,
    }
    if shooting == Shooting.MULTIPLE and not keep_intermediate_points:
        with pytest.raises(
            ValueError,
            match="shooting_type=Shooting.MULTIPLE and keep_intermediate_points=False cannot be used simultaneously."
            "When using multiple shooting, the intermediate points should be kept.",
        ):
            _ = sol.integrate(**opts)
        return

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

    sol_integrated = sol.integrate(**opts)
    shapes = (3, 3)
    states_shape_sum = 0
    time_shape_sum = 0
    for i in range(len(sol_integrated.states)):
        for key in sol_integrated.states[i].keys():
            states_shape_sum += np.shape(sol_integrated.states[i][key])[1]
    for t in sol_integrated.time:
        time_shape_sum += t.shape[0] * 2  # For q and qdot
    assert states_shape_sum == time_shape_sum

    decimal = 1 if integrator != SolutionIntegrator.OCP else 8
    for i in range(len(sol_integrated.states)):
        for k, key in enumerate(sol.states[i]):
            if integrator == SolutionIntegrator.OCP or shooting == Shooting.MULTIPLE:
                np.testing.assert_almost_equal(
                    sol_integrated.states[i][key][:, [0, -1]], sol.states[i][key][:, [0, -1]], decimal=decimal
                )

            if keep_intermediate_points:
                if shooting == Shooting.MULTIPLE:
                    assert sol_integrated.states[i][key].shape == (shapes[k], n_shooting[i] * 6 + 1)
                else:
                    assert sol_integrated.states[i][key].shape == (shapes[k], n_shooting[i] * 5 + 1)
            else:
                if integrator == SolutionIntegrator.OCP or shooting == Shooting.MULTIPLE:
                    np.testing.assert_almost_equal(sol_integrated.states[i][key], sol.states[i][key])
                assert sol_integrated.states[i][key].shape == (shapes[k], n_shooting[i] + 1)
            if ode_solver == OdeSolver.COLLOCATION:
                assert sol.states[i][key].shape == (shapes[k], n_shooting[i] * 5 + 1)
            else:
                assert sol.states[i][key].shape == (shapes[k], n_shooting[i] + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_integrated.controls


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("shooting", [Shooting.SINGLE, Shooting.MULTIPLE, Shooting.SINGLE_DISCONTINUOUS_PHASE])
@pytest.mark.parametrize("keep_intermediate_points", [True, False])
@pytest.mark.parametrize("integrator", [SolutionIntegrator.OCP, SolutionIntegrator.SCIPY_RK45])
def test_integrate_multiphase_merged(shooting, keep_intermediate_points, integrator, ode_solver, assume_phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        ode_solver=ode_solver(),
        assume_phase_dynamics=assume_phase_dynamics,
    )

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)

    opts = {
        "shooting_type": shooting,
        "keep_intermediate_points": keep_intermediate_points,
        "integrator": integrator,
    }

    if shooting == Shooting.MULTIPLE and not keep_intermediate_points:
        with pytest.raises(
            ValueError,
            match="shooting_type=Shooting.MULTIPLE and keep_intermediate_points=False cannot be used simultaneously."
            "When using multiple shooting, the intermediate points should be kept.",
        ):
            _ = sol.integrate(**opts)
        return

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

    opts["merge_phases"] = True

    n_shooting = [20, 30, 20]
    sol_integrated = sol.integrate(**opts)

    for key in sol_integrated.states.keys():
        assert np.shape(sol_integrated.states[key])[1] == np.shape(sol_integrated._time_vector)[1]

    shapes = (3, 3)
    decimal = 0 if integrator != SolutionIntegrator.OCP else 8
    for k, key in enumerate(sol.states[0]):
        expected = np.array([sol.states[0][key][:, 0], sol.states[-1][key][:, -1]]).T
        if integrator == SolutionIntegrator.OCP or shooting == Shooting.MULTIPLE:
            np.testing.assert_almost_equal(sol_integrated.states[key][:, [0, -1]], expected, decimal=decimal)

        if keep_intermediate_points:
            if shooting == Shooting.MULTIPLE:
                assert sol_integrated.states[key].shape == (shapes[k], sum(n_shooting) * 6 + 3 * 1)
            elif shooting == Shooting.SINGLE_DISCONTINUOUS_PHASE:
                assert sol_integrated.states[key].shape == (shapes[k], sum(n_shooting) * 5 + 3 * 1)
            else:
                assert sol_integrated.states[key].shape == (shapes[k], sum(n_shooting) * 5 + 1)
        else:
            # The interpolation prevents from comparing all points
            if integrator == SolutionIntegrator.OCP or shooting == Shooting.MULTIPLE:
                expected = np.concatenate(
                    (sol.states[0][key][:, 0:1], sol.states[-1][key][:, -1][:, np.newaxis]), axis=1
                )
                np.testing.assert_almost_equal(sol_integrated.states[key][:, [0, -1]], expected)

            if shooting == Shooting.SINGLE_DISCONTINUOUS_PHASE:
                assert sol_integrated.states[key].shape == (shapes[k], sum(n_shooting) + 3 * 1)
            else:
                assert sol_integrated.states[key].shape == (shapes[k], sum(n_shooting) + 1)

    for i in range(len(sol_integrated.states)):
        for k, key in enumerate(sol.states[i]):
            if ode_solver == OdeSolver.COLLOCATION:
                assert sol.states[i][key].shape == (shapes[k], n_shooting[i] * 5 + 1)
            else:
                assert sol.states[i][key].shape == (shapes[k], n_shooting[i] + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_integrated.controls
