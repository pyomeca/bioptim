import pytest

import numpy as np
from bioptim import Shooting

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


def test_interpolate_multiphases():
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_multiphase.py")

    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
    )

    sol = ocp.solve()
    n_frames = 100
    n_shooting = [20, 30, 20]
    sol_interp = sol.interpolate([n_frames, n_frames, n_frames])
    shapes = (6, 3, 3)
    for i, key in enumerate(sol.states[0]):
        np.testing.assert_almost_equal(sol_interp.states[i][key][:, [0, -1]], sol.states[i][key][:, [0, -1]])
        assert sol_interp.states[i][key].shape == (shapes[i], n_frames)
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


def test_integrate():
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
    sol_integrated = sol.integrate(shooting_type=Shooting.MULTIPLE)
    shapes = (4, 2, 2)

    for i, key in enumerate(sol.states):
        np.testing.assert_almost_equal(sol_integrated.states[key][:, [0, -1]], sol.states[key][:, [0, -1]])

        assert sol_integrated.states[key].shape == (shapes[i], n_shooting * 5 + 1)
        assert sol.states[key].shape == (shapes[i], n_shooting + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_integrated.controls


def test_integrate_single_shoot():
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
    sol_integrated = sol.integrate()
    shapes = (4, 2, 2)

    for i, key in enumerate(sol.states):
        np.testing.assert_almost_equal(sol_integrated.states[key][:, [0, -1]], sol.states[key][:, [0, -1]])

        assert sol_integrated.states[key].shape == (shapes[i], n_shooting * 5 + 1)
        assert sol.states[key].shape == (shapes[i], n_shooting + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_integrated.controls


@pytest.mark.parametrize("shooting", [Shooting.SINGLE_CONTINUOUS, Shooting.MULTIPLE, Shooting.SINGLE])
@pytest.mark.parametrize("merge", [False, True])
def test_integrate_non_continuous(shooting, merge):
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
    sol_integrated = sol.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS, continuous=False, merge_phases=merge)
    shapes = (4, 2, 2)

    for i, key in enumerate(sol.states):
        np.testing.assert_almost_equal(sol_integrated.states[key][:, [0, -1]], sol.states[key][:, [0, -1]])
        np.testing.assert_almost_equal(sol_integrated.states[key][:, [0, -2]], sol.states[key][:, [0, -1]])

        assert sol_integrated.states[key].shape == (shapes[i], n_shooting * (5 + 1) + 1)
        assert sol.states[key].shape == (shapes[i], n_shooting + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_integrated.controls


@pytest.mark.parametrize("shooting", [Shooting.SINGLE_CONTINUOUS, Shooting.MULTIPLE, Shooting.SINGLE])
def test_integrate_multiphase(shooting):
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_multiphase.py")

    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
    )

    sol = ocp.solve()
    n_shooting = [20, 30, 20]
    sol_integrated = sol.integrate(shooting_type=shooting)
    shapes = (6, 3, 3)

    for i in range(len(sol_integrated.states)):
        for k, key in enumerate(sol.states[i]):
            np.testing.assert_almost_equal(sol_integrated.states[i][key][:, [0, -1]], sol.states[i][key][:, [0, -1]])

            assert sol_integrated.states[i][key].shape == (shapes[k], n_shooting[i] * 5 + 1)
            assert sol.states[i][key].shape == (shapes[k], n_shooting[i] + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_integrated.controls


@pytest.mark.parametrize("shooting", [Shooting.SINGLE_CONTINUOUS, Shooting.MULTIPLE, Shooting.SINGLE])
def test_integrate_multiphase_merged(shooting):
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_multiphase.py")

    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
    )

    sol = ocp.solve()
    n_shooting = [20, 30, 20]
    sol_integrated = sol.integrate(shooting_type=shooting, merge_phases=True)
    shapes = (6, 3, 3)

    for k, key in enumerate(sol.states[0]):
        expected = np.array([sol.states[0][key][:, 0], sol.states[-1][key][:, -1]]).T
        np.testing.assert_almost_equal(sol_integrated.states[key][:, [0, -1]], expected)

        assert sol_integrated.states[key].shape == (shapes[k], sum(n_shooting) * 5 + 1)

    for i in range(len(sol_integrated.states)):
        for k, key in enumerate(sol.states[i]):
            assert sol.states[i][key].shape == (shapes[k], n_shooting[i] + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_integrated.controls


@pytest.mark.parametrize("shooting", [Shooting.SINGLE_CONTINUOUS, Shooting.MULTIPLE, Shooting.SINGLE])
def test_integrate_multiphase_non_continuous(shooting):
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_multiphase.py")

    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
    )

    sol = ocp.solve()
    n_shooting = [20, 30, 20]
    sol_integrated = sol.integrate(shooting_type=shooting, continuous=False)
    shapes = (6, 3, 3)

    for i in range(len(sol_integrated.states)):
        for k, key in enumerate(sol.states[i]):
            np.testing.assert_almost_equal(sol_integrated.states[i][key][:, [0, -1]], sol.states[i][key][:, [0, -1]])
            np.testing.assert_almost_equal(sol_integrated.states[i][key][:, [0, -2]], sol.states[i][key][:, [0, -1]])

            assert sol_integrated.states[i][key].shape == (shapes[k], n_shooting[i] * (5 + 1) + 1)
            assert sol.states[i][key].shape == (shapes[k], n_shooting[i] + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_integrated.controls


@pytest.mark.parametrize("shooting", [Shooting.SINGLE_CONTINUOUS, Shooting.MULTIPLE, Shooting.SINGLE])
def test_integrate_multiphase_merged_non_continuous(shooting):
    # Load pendulum
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_multiphase.py")

    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
    )

    sol = ocp.solve()
    n_shooting = [20, 30, 20]
    sol_integrated = sol.integrate(shooting_type=shooting, merge_phases=True, continuous=False)
    shapes = (6, 3, 3)

    for k, key in enumerate(sol.states[0]):
        expected = np.array([sol.states[0][key][:, 0], sol.states[-1][key][:, -1]]).T
        np.testing.assert_almost_equal(sol_integrated.states[key][:, [0, -1]], expected)
        np.testing.assert_almost_equal(sol_integrated.states[key][:, [0, -2]], expected)

        assert sol_integrated.states[key].shape == (shapes[k], sum(n_shooting) * (5 + 1) + 1)

    for i in range(len(sol_integrated.states)):
        for k, key in enumerate(sol.states[i]):
            assert sol.states[i][key].shape == (shapes[k], n_shooting[i] + 1)

    with pytest.raises(
        RuntimeError,
        match="There is no controls in the solution. This may happen in previously "
        "integrated and interpolated structure",
    ):
        _ = sol_integrated.controls
