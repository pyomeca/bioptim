import pytest
import importlib.util
from pathlib import Path

import numpy as np

from bioptim import Data, InterpolationType, Simulate, InitialGuess

# TODO: Add negative test for sizes


def test_initial_guess_constant():
    nb_elements = 6
    nb_shoot = 10

    init_val = np.random.random(
        nb_elements,
    )
    init = InitialGuess(init_val, interpolation=InterpolationType.CONSTANT)
    init.check_and_adjust_dimensions(nb_elements, nb_shoot)
    expected_val = init_val
    for i in range(nb_shoot):
        np.testing.assert_almost_equal(init.init.evaluate_at(i), expected_val)


def test_initial_guess_constant_with_first_and_last_different():
    nb_elements = 6
    nb_shoot = 10

    init_val = np.random.random((nb_elements, 3))

    init = InitialGuess(init_val, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    init.check_and_adjust_dimensions(nb_elements, nb_shoot)
    for i in range(nb_shoot + 1):
        if i == 0:
            expected_val = init_val[:, 0]
        elif i == nb_shoot:
            expected_val = init_val[:, 2]
        else:
            expected_val = init_val[:, 1]
        np.testing.assert_almost_equal(init.init.evaluate_at(i), expected_val)


def test_initial_guess_linear():
    nb_elements = 6
    nb_shoot = 10

    init_val = np.random.random((nb_elements, 2))

    init = InitialGuess(init_val, interpolation=InterpolationType.LINEAR)
    init.check_and_adjust_dimensions(nb_elements, nb_shoot)
    for i in range(nb_shoot + 1):
        expected_val = init_val[:, 0] + (init_val[:, 1] - init_val[:, 0]) * i / nb_shoot
        np.testing.assert_almost_equal(init.init.evaluate_at(i), expected_val)


def test_initial_guess_each_frame():
    nb_elements = 6
    nb_shoot = 10

    init_val = np.random.random((nb_elements, nb_shoot + 1))

    init = InitialGuess(init_val, interpolation=InterpolationType.EACH_FRAME)
    init.check_and_adjust_dimensions(nb_elements, nb_shoot)
    for i in range(nb_shoot + 1):
        expected_val = init_val[:, i]
        np.testing.assert_almost_equal(init.init.evaluate_at(i), expected_val)


def test_initial_guess_spline():
    nb_shoot = 10
    spline_time = np.hstack((0.0, 1.0, 2.2, 6.0))
    init_val = np.array(
        [
            [0.5, 0.6, 0.2, 0.8],
            [0.4, 0.6, 0.8, 0.2],
            [0.0, 0.3, 0.2, 0.5],
            [0.5, 0.2, 0.9, 0.4],
            [0.5, 0.6, 0.2, 0.8],
            [0.5, 0.6, 0.2, 0.8],
        ]
    )
    nb_elements = init_val.shape[0]

    # Raise if time is not sent
    with pytest.raises(RuntimeError):
        InitialGuess(init_val, interpolation=InterpolationType.SPLINE)

    init = InitialGuess(init_val, t=spline_time, interpolation=InterpolationType.SPLINE)
    init.check_and_adjust_dimensions(nb_elements, nb_shoot)

    time_to_test = [0, nb_shoot // 3, nb_shoot // 2, nb_shoot]
    expected_matrix = np.array(
        [
            [0.5, 0.4, 0.0, 0.5, 0.5, 0.5],
            [0.33333333, 0.73333333, 0.23333333, 0.66666667, 0.33333333, 0.33333333],
            [0.32631579, 0.67368421, 0.26315789, 0.79473684, 0.32631579, 0.32631579],
            [0.8, 0.2, 0.5, 0.4, 0.8, 0.8],
        ]
    ).T
    for i, t in enumerate(time_to_test):
        expected_val = expected_matrix[:, i]
        np.testing.assert_almost_equal(init.init.evaluate_at(t), expected_val)


def test_initial_guess_update():
    # Load pendulum
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "pendulum", str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/pendulum_min_time_Mayer.py"
    )
    pendulum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pendulum)

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/pendulum.bioMod",
        final_time=2,
        number_shooting_points=10,
    )

    np.testing.assert_almost_equal(ocp.nlp[0].x_init.init, np.zeros((4, 1)))
    np.testing.assert_almost_equal(ocp.nlp[0].u_init.init, np.zeros((2, 1)))
    np.testing.assert_almost_equal(ocp.param_to_optimize["time"].initial_guess.init[0, 0], 2)
    np.testing.assert_almost_equal(ocp.V_init.init, np.concatenate(([[2]], np.zeros((4 * 11 + 2 * 10, 1)))))

    wrong_new_x_init = InitialGuess([1] * 6)
    new_x_init = InitialGuess([1] * 4)
    wrong_new_u_init = InitialGuess([3] * 4)
    new_u_init = InitialGuess([3] * 2)
    new_time_init = InitialGuess([4])

    # No name for the param
    with pytest.raises(ValueError, match="update_initial_guess must specify a name for the parameters"):
        ocp.update_initial_guess(new_x_init, new_u_init, new_time_init)

    new_time_init.name = "dumb name"
    with pytest.raises(ValueError, match="update_initial_guess cannot declare new parameters"):
        ocp.update_initial_guess(new_x_init, new_u_init, new_time_init)

    new_time_init.name = "time"
    with pytest.raises(RuntimeError):
        ocp.update_initial_guess(new_x_init, wrong_new_u_init, new_time_init)
    with pytest.raises(RuntimeError):
        ocp.update_initial_guess(wrong_new_x_init, wrong_new_u_init, new_time_init)

    ocp.update_initial_guess(new_x_init, new_u_init, new_time_init)

    np.testing.assert_almost_equal(ocp.nlp[0].x_init.init, np.ones((4, 1)))
    np.testing.assert_almost_equal(ocp.nlp[0].u_init.init, np.ones((2, 1)) * 3)
    np.testing.assert_almost_equal(ocp.param_to_optimize["time"].initial_guess.init[0, 0], 4)
    np.testing.assert_almost_equal(ocp.V_init.init, np.array([[4] + [1, 1, 1, 1, 3, 3] * 10 + [1, 1, 1, 1]]).T)


def test_initial_guess_custom():
    nb_elements = 6
    nb_shoot = 10

    def custom_bound_func(current_shooting, val, total_shooting):
        # Linear interpolation created with custom bound function
        return val[:, 0] + (val[:, 1] - val[:, 0]) * current_shooting / total_shooting

    init_val = np.random.random((nb_elements, 2))

    init = InitialGuess(
        custom_bound_func,
        interpolation=InterpolationType.CUSTOM,
        val=init_val,
        total_shooting=nb_shoot,
    )
    init.check_and_adjust_dimensions(nb_elements, nb_shoot)
    for i in range(nb_shoot + 1):
        expected_val = init_val[:, 0] + (init_val[:, 1] - init_val[:, 0]) * i / nb_shoot
        np.testing.assert_almost_equal(init.init.evaluate_at(i), expected_val)


def test_simulate_from_initial_multiple_shoot():
    # Load pendulum
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "pendulum", str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.py"
    )
    pendulum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pendulum)

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
        final_time=2,
        number_shooting_points=10,
        nb_threads=4,
    )

    X = InitialGuess([-1, -2, 1, 0.5])
    U = InitialGuess(np.array([[-0.1, 0], [1, 2]]).T, interpolation=InterpolationType.LINEAR)

    sol_simulate_multiple_shooting = Simulate.from_controls_and_initial_states(ocp, X, U, single_shoot=False)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol_simulate_multiple_shooting["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((-1.0, -2.0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-0.76453657, -1.78061188)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((1.0, 0.5)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((1.22338503, 1.68361549)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-0.1, 0.0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((1.0, 2.0)))


def test_simulate_from_initial_single_shoot():
    # Load pendulum
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "pendulum", str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.py"
    )
    pendulum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pendulum)

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
        final_time=2,
        number_shooting_points=10,
        nb_threads=4,
    )

    X = InitialGuess([-1, -2, 1, 0.5])
    U = InitialGuess(np.array([[-0.1, 0], [1, 2]]).T, interpolation=InterpolationType.LINEAR)

    sol_simulate_single_shooting = Simulate.from_controls_and_initial_states(ocp, X, U, single_shoot=True)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol_simulate_single_shooting["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((-1.0, -2.0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-0.59371229, 2.09731719)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((1.0, 0.5)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((1.38153013, -0.60425128)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-0.1, 0.0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((1.0, 2.0)))
