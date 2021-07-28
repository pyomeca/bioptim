import pytest

import numpy as np
from bioptim import Shooting, OdeSolver

from .utils import TestUtils


def test_merge_phases_one_phase():
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    pendulum = TestUtils.load_module(bioptim_folder + "/examples/getting_started/pendulum.py")

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod",
        final_time=2,
        n_shooting=10,
    )

    sol = ocp.solve()
    sol_merged = sol.merge_phases()
    for key in sol.states:
        np.testing.assert_almost_equal(sol_merged.states[key], sol.states[key])
    for key in sol.controls:
        np.testing.assert_almost_equal(sol_merged.controls[key], sol.controls[key])


def test_merge_phases_multi_phase():
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_multiphase.py")

    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
    )

    sol = ocp.solve()
    sol_merged = sol.merge_phases()

    for key in sol.states[0]:
        expected = np.concatenate([s[key][:, :-1] for s in sol.states], axis=1)
        expected = np.concatenate((expected, sol.states[-1][key][:, -1][:, np.newaxis]), axis=1)

        np.testing.assert_almost_equal(sol_merged.states[key], expected)

    for key in sol.controls[0]:
        expected = np.concatenate([s[key][:, :-1] for s in sol.controls], axis=1)
        expected = np.concatenate((expected, sol.controls[-1][key][:, -1][:, np.newaxis]), axis=1)

        np.testing.assert_almost_equal(sol_merged.controls[key], expected)


def test_interpolate():
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    pendulum = TestUtils.load_module(bioptim_folder + "/examples/getting_started/pendulum.py")
    n_shooting = 10

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod",
        final_time=2,
        n_shooting=n_shooting,
    )

    sol = ocp.solve()
    n_frames = 100
    sol_interp = sol.interpolate(n_frames)
    sol_interp_list = sol.interpolate([n_frames])
    shapes = (4, 2, 2)
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


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
def test_interpolate_multiphases(ode_solver):
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_multiphase.py")

    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod", ode_solver=ode_solver()
    )

    sol = ocp.solve()
    n_frames = 100
    n_shooting = [20, 30, 20]
    sol_interp = sol.interpolate([n_frames, n_frames, n_frames])
    shapes = (6, 3, 3)

    decimal = 2 if ode_solver == OdeSolver.COLLOCATION else 8
    for i, key in enumerate(sol.states[0]):
        np.testing.assert_almost_equal(sol_interp.states[i][key][:, [0, -1]], sol.states[i][key][:, [0, -1]], decimal=decimal)
        assert sol_interp.states[i][key].shape == (shapes[i], n_frames)
        if ode_solver == OdeSolver.COLLOCATION:
            assert sol.states[i][key].shape == (shapes[i], n_shooting[i] * 5 + 1)
        else:
            assert sol.states[i][key].shape == (shapes[i], n_shooting[i]+ 1)

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


def test_interpolate_multiphases_merge_phase():
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_multiphase.py")

    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
    )

    sol = ocp.solve()
    n_frames = 100
    n_shooting = [20, 30, 20]
    sol_interp = sol.interpolate(n_frames)
    shapes = (6, 3, 3)

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


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("use_scipy", [False, True])
def test_integrate(use_scipy, ode_solver):
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    pendulum = TestUtils.load_module(bioptim_folder + "/examples/getting_started/pendulum.py")
    n_shooting = 20

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod",
        final_time=3,
        n_shooting=n_shooting,
        ode_solver=ode_solver(),
    )

    sol = ocp.solve()

    opts = {"shooting_type": Shooting.MULTIPLE, "keep_intermediate_points": False, "use_scipy_integrator": use_scipy}
    with pytest.raises(
        ValueError,
        match="Shooting.MULTIPLE and keep_intermediate_points=False "
        "cannot be used simultaneously since it would do nothing",
    ):
        _ = sol.integrate(**opts)

    opts["keep_intermediate_points"] = True
    if ode_solver == OdeSolver.COLLOCATION and not use_scipy:
        with pytest.raises(RuntimeError, match="Integration with direct collocation must be not continuous"):
            sol.integrate(**opts)
        return

    sol_integrated = sol.integrate(**opts)
    shapes = (4, 2, 2)

    decimal = 5 if use_scipy else 8
    for i, key in enumerate(sol.states):
        np.testing.assert_almost_equal(
            sol_integrated.states[key][:, [0, -1]], sol.states[key][:, [0, -1]], decimal=decimal
        )

        assert sol_integrated.states[key].shape == (shapes[i], n_shooting * 5 + 1)
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


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("keep_intermediate_points", [False, True])
@pytest.mark.parametrize("use_scipy", [False, True])
def test_integrate_single_shoot(keep_intermediate_points, use_scipy, ode_solver):
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    pendulum = TestUtils.load_module(bioptim_folder + "/examples/getting_started/pendulum.py")
    n_shooting = 20

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod",
        final_time=3,
        n_shooting=n_shooting,
        ode_solver=ode_solver(),
    )

    sol = ocp.solve()

    opts = {"keep_intermediate_points": keep_intermediate_points, "use_scipy_integrator": use_scipy}
    if ode_solver == OdeSolver.COLLOCATION and not use_scipy:
        with pytest.raises(RuntimeError, match="Integration with direct collocation must be not continuous"):
            sol.integrate(**opts)
        return

    sol_integrated = sol.integrate(**opts)
    shapes = (4, 2, 2)

    decimal = 1 if use_scipy else 8
    for i, key in enumerate(sol.states):
        np.testing.assert_almost_equal(
            sol_integrated.states[key][:, [0, -1]], sol.states[key][:, [0, -1]], decimal=decimal
        )

        if keep_intermediate_points or (ode_solver == OdeSolver.COLLOCATION and not use_scipy):
            assert sol_integrated.states[key].shape == (shapes[i], n_shooting * 5 + 1)
        else:
            if not (ode_solver == OdeSolver.COLLOCATION and use_scipy):
                np.testing.assert_almost_equal(sol_integrated.states[key], sol.states[key], decimal=decimal)
            assert sol_integrated.states[key].shape == (shapes[i], n_shooting + 1)

        if ode_solver == OdeSolver.COLLOCATION and use_scipy:
            assert sol.states[key].shape == (shapes[i], n_shooting * 5 + 1)
        else:
            assert sol.states[key].shape == (shapes[i], n_shooting + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_integrated.controls


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("shooting", [Shooting.SINGLE_CONTINUOUS, Shooting.MULTIPLE, Shooting.SINGLE])
@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("use_scipy", [False, True])
def test_integrate_non_continuous(shooting, merge, use_scipy, ode_solver):
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    pendulum = TestUtils.load_module(bioptim_folder + "/examples/getting_started/pendulum.py")
    n_shooting = 20

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod",
        final_time=3,
        n_shooting=n_shooting,
        ode_solver=ode_solver(),
    )

    sol = ocp.solve()

    opts = {
        "shooting_type": shooting,
        "continuous": False,
        "keep_intermediate_points": False,
        "use_scipy_integrator": use_scipy,
    }
    if shooting == Shooting.MULTIPLE:
        with pytest.raises(
            ValueError,
            match="Shooting.MULTIPLE and keep_intermediate_points=False cannot be used simultaneously since it would do nothing",
        ):
            _ = sol.integrate(**opts)

    opts["keep_intermediate_points"] = True
    opts["merge_phases"] = merge
    if ode_solver == OdeSolver.COLLOCATION and shooting != Shooting.MULTIPLE and not use_scipy:
        with pytest.raises(
            RuntimeError,
            match="Integration with direct collocation must using shooting_type=Shooting.MULTIPLE",
        ):
            sol.integrate(**opts)
        return

    sol_integrated = sol.integrate(**opts)
    shapes = (4, 2, 2)

    decimal = 1 if use_scipy or ode_solver == OdeSolver.COLLOCATION else 8
    for i, key in enumerate(sol.states):
        np.testing.assert_almost_equal(
            sol_integrated.states[key][:, [0, -1]], sol.states[key][:, [0, -1]], decimal=decimal
        )
        np.testing.assert_almost_equal(
            sol_integrated.states[key][:, [0, -2]], sol.states[key][:, [0, -1]], decimal=decimal
        )

        if ode_solver == OdeSolver.COLLOCATION:
            if use_scipy:
                assert sol_integrated.states[key].shape == (shapes[i], n_shooting * (5 + 1) + 1)
            else:
                assert sol_integrated.states[key].shape == (shapes[i], n_shooting * (4 + 1) + 1)
            assert sol.states[key].shape == (shapes[i], n_shooting * 5 + 1)
        else:
            assert sol_integrated.states[key].shape == (shapes[i], n_shooting * (5 + 1) + 1)
            assert sol.states[key].shape == (shapes[i], n_shooting + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_integrated.controls


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("shooting", [Shooting.SINGLE_CONTINUOUS, Shooting.MULTIPLE, Shooting.SINGLE])
@pytest.mark.parametrize("keep_intermediate_points", [True, False])
@pytest.mark.parametrize("use_scipy", [False, True])
def test_integrate_multiphase(shooting, keep_intermediate_points, use_scipy, ode_solver):
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_multiphase.py")

    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod", ode_solver=ode_solver()
    )

    sol = ocp.solve()
    n_shooting = [20, 30, 20]

    opts = {
        "shooting_type": shooting,
        "continuous": False,
        "keep_intermediate_points": keep_intermediate_points,
        "use_scipy_integrator": use_scipy,
    }
    if ode_solver == OdeSolver.COLLOCATION and not use_scipy:
        if shooting != Shooting.MULTIPLE:
            with pytest.raises(
                RuntimeError, match="Integration with direct collocation must using shooting_type=Shooting.MULTIPLE"
            ):
                _ = sol.integrate(**opts)
            return

    if shooting == Shooting.MULTIPLE and not keep_intermediate_points:
        with pytest.raises(
            ValueError,
            match="Shooting.MULTIPLE and keep_intermediate_points=False cannot be used simultaneously since it would do nothing",
        ):
            _ = sol.integrate(**opts)
        return

    opts["continuous"] = True
    if ode_solver == OdeSolver.COLLOCATION and not use_scipy:
        with pytest.raises(
            RuntimeError,
            match="Integration with direct collocation must be not continuous",
        ):
            sol.integrate(**opts)
        return

    sol_integrated = sol.integrate(**opts)
    shapes = (6, 3, 3)

    decimal = 1 if use_scipy else 8
    for i in range(len(sol_integrated.states)):
        for k, key in enumerate(sol.states[i]):
            if not use_scipy or shooting == Shooting.MULTIPLE:
                np.testing.assert_almost_equal(
                    sol_integrated.states[i][key][:, [0, -1]], sol.states[i][key][:, [0, -1]], decimal=decimal
                )

            if keep_intermediate_points:
                assert sol_integrated.states[i][key].shape == (shapes[k], n_shooting[i] * 5 + 1)
            else:
                if not use_scipy or shooting == Shooting.MULTIPLE:
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


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("shooting", [Shooting.SINGLE_CONTINUOUS, Shooting.MULTIPLE, Shooting.SINGLE])
@pytest.mark.parametrize("keep_intermediate_points", [True, False])
@pytest.mark.parametrize("use_scipy", [False, True])
def test_integrate_multiphase_merged(shooting, keep_intermediate_points, use_scipy, ode_solver):
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_multiphase.py")

    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
        ode_solver=ode_solver(),
    )

    sol = ocp.solve()

    opts = {
        "shooting_type": shooting,
        "continuous": False,
        "keep_intermediate_points": keep_intermediate_points,
        "use_scipy_integrator": use_scipy,
    }
    if ode_solver == OdeSolver.COLLOCATION and not use_scipy:
        if shooting != Shooting.MULTIPLE:
            with pytest.raises(
                RuntimeError, match="Integration with direct collocation must using shooting_type=Shooting.MULTIPLE"
            ):
                _ = sol.integrate(**opts)
            return

    if shooting == Shooting.MULTIPLE and not keep_intermediate_points:
        with pytest.raises(
            ValueError,
            match="Shooting.MULTIPLE and keep_intermediate_points=False cannot be used simultaneously since it would do nothing",
        ):
            _ = sol.integrate(**opts)
        return

    opts["merge_phases"] = True
    opts["continuous"] = True
    if ode_solver == OdeSolver.COLLOCATION and not use_scipy:
        with pytest.raises(
            RuntimeError,
            match="Integration with direct collocation must be not continuous",
        ):
            sol.integrate(**opts)
        return

    n_shooting = [20, 30, 20]
    sol_integrated = sol.integrate(**opts)
    shapes = (6, 3, 3)

    decimal = 0 if use_scipy else 8
    for k, key in enumerate(sol.states[0]):
        expected = np.array([sol.states[0][key][:, 0], sol.states[-1][key][:, -1]]).T
        if not use_scipy or shooting == Shooting.MULTIPLE:
            np.testing.assert_almost_equal(sol_integrated.states[key][:, [0, -1]], expected, decimal=decimal)

        if keep_intermediate_points:
            assert sol_integrated.states[key].shape == (shapes[k], sum(n_shooting) * 5 + 1)
        else:
            # The interpolation prevents from comparing all points
            if not use_scipy or shooting == Shooting.MULTIPLE:
                expected = np.concatenate(
                    (sol.states[0][key][:, 0:1], sol.states[-1][key][:, -1][:, np.newaxis]), axis=1
                )
                np.testing.assert_almost_equal(sol_integrated.states[key][:, [0, -1]], expected)
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


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("shooting", [Shooting.SINGLE_CONTINUOUS, Shooting.MULTIPLE, Shooting.SINGLE])
@pytest.mark.parametrize("use_scipy", [False, True])
def test_integrate_multiphase_non_continuous(shooting, use_scipy, ode_solver):
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_multiphase.py")

    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod", ode_solver=ode_solver()
    )

    sol = ocp.solve()
    n_shooting = [20, 30, 20]

    opts = {
        "shooting_type": shooting,
        "continuous": False,
        "keep_intermediate_points": True,
        "use_scipy_integrator": use_scipy,
    }
    if ode_solver == OdeSolver.COLLOCATION and not use_scipy:
        if shooting != Shooting.MULTIPLE:
            with pytest.raises(
                RuntimeError, match="Integration with direct collocation must using shooting_type=Shooting.MULTIPLE"
            ):
                _ = sol.integrate(**opts)
            return

    sol_integrated = sol.integrate(**opts)
    shapes = (6, 3, 3)

    decimal = 1 if use_scipy or ode_solver == OdeSolver.COLLOCATION else 8
    for i in range(len(sol_integrated.states)):
        for k, key in enumerate(sol.states[i]):
            if not use_scipy or shooting == Shooting.MULTIPLE:
                np.testing.assert_almost_equal(
                    sol_integrated.states[i][key][:, [0, -1]], sol.states[i][key][:, [0, -1]], decimal=decimal
                )
                np.testing.assert_almost_equal(
                    sol_integrated.states[i][key][:, [0, -2]], sol.states[i][key][:, [0, -1]], decimal=decimal
                )

            if ode_solver == OdeSolver.COLLOCATION and not use_scipy:
                assert sol_integrated.states[i][key].shape == (shapes[k], n_shooting[i] * (4 + 1) + 1)
            else:
                assert sol_integrated.states[i][key].shape == (shapes[k], n_shooting[i] * (5 + 1) + 1)

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


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("shooting", [Shooting.SINGLE_CONTINUOUS, Shooting.MULTIPLE, Shooting.SINGLE])
@pytest.mark.parametrize("use_scipy", [False, True])
def test_integrate_multiphase_merged_non_continuous(shooting, use_scipy, ode_solver):
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_multiphase.py")

    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod", ode_solver=ode_solver()
    )

    sol = ocp.solve()

    opts = {
        "shooting_type": shooting,
        "continuous": False,
        "keep_intermediate_points": False,
        "use_scipy_integrator": use_scipy,
    }
    if shooting == Shooting.MULTIPLE:
        with pytest.raises(
            ValueError,
            match="Shooting.MULTIPLE and keep_intermediate_points=False cannot be used simultaneously since it would do nothing",
        ):
            _ = sol.integrate(**opts)
    else:
        if ode_solver == OdeSolver.COLLOCATION and not use_scipy:
            with pytest.raises(
                RuntimeError,
                match="Integration with direct collocation must using shooting_type=Shooting.MULTIPLE",
            ):
                _ = sol.integrate(**opts)
            return

    opts["merge_phases"] = True
    opts["keep_intermediate_points"] = True
    n_shooting = [20, 30, 20]
    sol_integrated = sol.integrate(**opts)
    shapes = (6, 3, 3)

    decimal = 0 if use_scipy or ode_solver == OdeSolver.COLLOCATION else 8
    steps = 4 if not use_scipy and ode_solver == OdeSolver.COLLOCATION else 5
    for k, key in enumerate(sol.states[0]):
        expected = np.array([sol.states[0][key][:, 0], sol.states[-1][key][:, -1]]).T
        if not use_scipy or shooting == Shooting.MULTIPLE:
            np.testing.assert_almost_equal(sol_integrated.states[key][:, [0, -1]], expected, decimal=decimal)
            np.testing.assert_almost_equal(sol_integrated.states[key][:, [0, -2]], expected, decimal=decimal)

        assert sol_integrated.states[key].shape == (shapes[k], sum(n_shooting) * (steps + 1) + 1)

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
