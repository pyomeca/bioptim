import os
import pytest

import numpy as np
from bioptim import (
    InterpolationType,
    InitialGuess,
    Solution,
    Shooting,
    SolutionIntegrator,
    BiorbdModel,
    Objective,
    Dynamics,
    DynamicsFcn,
    Bounds,
    ObjectiveFcn,
    OptimalControlProgram,
    InitialGuessList,
)

# TODO: Add negative test for sizes


def test_initial_guess_constant():
    n_elements = 6
    n_shoot = 10

    init_val = np.random.random(
        n_elements,
    )
    init = InitialGuess(init_val, interpolation=InterpolationType.CONSTANT)
    init.check_and_adjust_dimensions(n_elements, n_shoot)
    expected_val = init_val
    for i in range(n_shoot):
        np.testing.assert_almost_equal(init.init.evaluate_at(i), expected_val)


def test_initial_guess_constant_with_first_and_last_different():
    n_elements = 6
    n_shoot = 10

    init_val = np.random.random((n_elements, 3))

    init = InitialGuess(init_val, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    init.check_and_adjust_dimensions(n_elements, n_shoot)
    for i in range(n_shoot + 1):
        if i == 0:
            expected_val = init_val[:, 0]
        elif i == n_shoot:
            expected_val = init_val[:, 2]
        else:
            expected_val = init_val[:, 1]
        np.testing.assert_almost_equal(init.init.evaluate_at(i), expected_val)


def test_initial_guess_linear():
    n_elements = 6
    n_shoot = 10

    init_val = np.random.random((n_elements, 2))

    init = InitialGuess(init_val, interpolation=InterpolationType.LINEAR)
    init.check_and_adjust_dimensions(n_elements, n_shoot)
    for i in range(n_shoot + 1):
        expected_val = init_val[:, 0] + (init_val[:, 1] - init_val[:, 0]) * i / n_shoot
        np.testing.assert_almost_equal(init.init.evaluate_at(i), expected_val)


def test_initial_guess_each_frame():
    n_elements = 6
    n_shoot = 10

    init_val = np.random.random((n_elements, n_shoot + 1))

    init = InitialGuess(init_val, interpolation=InterpolationType.EACH_FRAME)
    init.check_and_adjust_dimensions(n_elements, n_shoot)
    for i in range(n_shoot + 1):
        expected_val = init_val[:, i]
        np.testing.assert_almost_equal(init.init.evaluate_at(i), expected_val)


def test_initial_guess_spline():
    n_shoot = 10
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
    n_elements = init_val.shape[0]

    # Raise if time is not sent
    with pytest.raises(RuntimeError):
        InitialGuess(init_val, interpolation=InterpolationType.SPLINE)

    init = InitialGuess(init_val, t=spline_time, interpolation=InterpolationType.SPLINE)
    init.check_and_adjust_dimensions(n_elements, n_shoot)

    time_to_test = [0, n_shoot // 3, n_shoot // 2, n_shoot]
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


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_initial_guess_update(assume_phase_dynamics):
    # Load pendulum
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Mayer as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=10,
        assume_phase_dynamics=assume_phase_dynamics,
    )

    np.testing.assert_almost_equal(ocp.nlp[0].x_init.init, np.zeros((4, 1)))
    np.testing.assert_almost_equal(ocp.nlp[0].u_init.init, np.zeros((2, 1)))
    idx = ocp.v.parameters_in_list.index("time")
    np.testing.assert_almost_equal(ocp.v.parameters_in_list[idx].initial_guess.init[0, 0], 2)
    np.testing.assert_almost_equal(ocp.v.init.init, np.concatenate((np.zeros((4 * 11 + 2 * 10, 1)), [[2]])))

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
    idx = ocp.v.parameters_in_list.index("time")
    np.testing.assert_almost_equal(ocp.v.parameters_in_list[idx].initial_guess.init[0, 0], 4)
    np.testing.assert_almost_equal(ocp.v.init.init, np.array([[1, 1, 1, 1] * 11 + [3, 3] * 10 + [4]]).T)


def test_initial_guess_custom():
    n_elements = 6
    n_shoot = 10

    def custom_bound_func(current_shooting, val, total_shooting):
        # Linear interpolation created with custom bound function
        return val[:, 0] + (val[:, 1] - val[:, 0]) * current_shooting / total_shooting

    init_val = np.random.random((n_elements, 2))

    init = InitialGuess(
        custom_bound_func,
        interpolation=InterpolationType.CUSTOM,
        val=init_val,
        total_shooting=n_shoot,
    )
    init.check_and_adjust_dimensions(n_elements, n_shoot)
    for i in range(n_shoot + 1):
        expected_val = init_val[:, 0] + (init_val[:, 1] - init_val[:, 0]) * i / n_shoot
        np.testing.assert_almost_equal(init.init.evaluate_at(i), expected_val)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_simulate_from_initial_multiple_shoot(assume_phase_dynamics):
    from bioptim.examples.getting_started import example_save_and_load as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=10,
        n_threads=4,
        assume_phase_dynamics=assume_phase_dynamics,
    )

    X = InitialGuess([-1, -2, 1, 0.5])
    U = InitialGuess(np.array([[-0.1, 0], [1, 2]]).T, interpolation=InterpolationType.LINEAR)

    sol = Solution(ocp, [X, U])
    controls = sol.controls
    sol = sol.integrate(
        shooting_type=Shooting.MULTIPLE, keep_intermediate_points=True, integrator=SolutionIntegrator.OCP
    )
    states = sol.states

    # Check some of the results
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((-1.0, -2.0)))
    np.testing.assert_almost_equal(q[:, -2], np.array((-0.7553692, -1.6579819)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((1.0, 0.5)))
    np.testing.assert_almost_equal(qdot[:, -2], np.array((1.05240919, 2.864199)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-0.1, 0.0)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.89, 1.8)))


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_simulate_from_initial_single_shoot(assume_phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import example_save_and_load as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=10,
        n_threads=4,
        assume_phase_dynamics=assume_phase_dynamics,
    )

    X = InitialGuess([-1, -2, 0.1, 0.2])
    U = InitialGuess(np.array([[-0.1, 0], [1, 2]]).T, interpolation=InterpolationType.LINEAR)

    sol = Solution(ocp, [X, U])
    controls = sol.controls
    sol = sol.integrate(shooting_type=Shooting.SINGLE, keep_intermediate_points=True, integrator=SolutionIntegrator.OCP)

    # Check some of the results
    states = sol.states
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((-1.0, -2.0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-0.33208579, 0.06094747)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0.1, 0.2)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-4.43192682, 6.38146735)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-0.1, 0.0)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.89, 1.8)))


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_initial_guess_error_messages(assume_phase_dynamics):
    """
    This tests that the error messages are properly raised. The OCP is adapted from the getting_started/pendulum.py example.
    """
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    biorbd_model_path = bioptim_folder + "/models/pendulum.bioMod"
    bio_model = BiorbdModel(biorbd_model_path)
    n_q = bio_model.nb_q

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Define control path constraint
    u_bounds = Bounds([-100] * n_q, [100] * n_q)
    u_bounds[1, :] = 0  # Prevent the model from actively rotate

    x_init = InitialGuess([0] * n_q * 2)
    u_init = InitialGuess([0] * n_q)

    # check the error messages
    with pytest.raises(RuntimeError, match="Please provide an x_init of type InitialGuess or InitialGuessList."):
        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting=5,
            phase_time=1,
            x_init=None,
            u_init=u_init,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            assume_phase_dynamics=assume_phase_dynamics,
        )
    with pytest.raises(RuntimeError, match="Please provide an u_init of type InitialGuess or InitialGuessList."):
        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting=5,
            phase_time=1,
            x_init=x_init,
            u_init=None,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            assume_phase_dynamics=assume_phase_dynamics,
        )

    with pytest.raises(
        RuntimeError,
        match="You must please declare an initial guess for the states (x_init). Here, the InitialGuessList is empty.",
    ):
        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting=5,
            phase_time=1,
            x_init=InitialGuessList(),
            u_init=u_init,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            assume_phase_dynamics=assume_phase_dynamics,
        )

    with pytest.raises(
        RuntimeError,
        match="You must please declare an initial guess for the controls u_init. Here, the InitialGuessList is empty.",
    ):
        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting=5,
            phase_time=1,
            x_init=x_init,
            u_init=InitialGuessList(),
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            assume_phase_dynamics=assume_phase_dynamics,
        )
