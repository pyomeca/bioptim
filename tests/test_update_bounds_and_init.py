import pytest
import numpy as np
import biorbd_casadi as biorbd
from casadi import MX
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    Bounds,
    ParameterList,
    InterpolationType,
    InitialGuess,
    Objective,
    ObjectiveFcn,
    QAndQDotBounds,
    NoisedInitialGuess,
)

from .utils import TestUtils


def test_double_update_bounds_and_init():
    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model = biorbd.Model(bioptim_folder + "/examples/track/models/cube_and_line.bioMod")
    nq = biorbd_model.nbQ()
    ns = 10

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, ns, 1.0)

    x_bounds = Bounds(-np.ones((nq * 2, 1)), np.ones((nq * 2, 1)))
    u_bounds = Bounds(-2.0 * np.ones((nq, 1)), 2.0 * np.ones((nq, 1)))
    ocp.update_bounds(x_bounds, u_bounds)

    expected = np.array([[-1] * (nq * 2) * (ns + 1) + [-2] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.bounds.min, expected)
    expected = np.array([[1] * (nq * 2) * (ns + 1) + [2] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.bounds.max, expected)

    x_init = InitialGuess(0.5 * np.ones((nq * 2, 1)))
    u_init = InitialGuess(-0.5 * np.ones((nq, 1)))
    ocp.update_initial_guess(x_init, u_init)
    expected = np.array([[0.5] * (nq * 2) * (ns + 1) + [-0.5] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.init.init, expected)

    x_bounds = Bounds(-2.0 * np.ones((nq * 2, 1)), 2.0 * np.ones((nq * 2, 1)))
    u_bounds = Bounds(-4.0 * np.ones((nq, 1)), 4.0 * np.ones((nq, 1)))
    ocp.update_bounds(x_bounds=x_bounds)
    ocp.update_bounds(u_bounds=u_bounds)

    expected = np.array([[-2] * (nq * 2) * (ns + 1) + [-4] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.bounds.min, expected)
    expected = np.array([[2] * (nq * 2) * (ns + 1) + [4] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.bounds.max, expected)

    x_init = InitialGuess(0.25 * np.ones((nq * 2, 1)))
    u_init = InitialGuess(-0.25 * np.ones((nq, 1)))
    ocp.update_initial_guess(x_init, u_init)
    expected = np.array([[0.25] * (nq * 2) * (ns + 1) + [-0.25] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.init.init, expected)

    with pytest.raises(RuntimeError, match="x_init should be built from a InitialGuess or InitialGuessList"):
        ocp.update_initial_guess(x_bounds, u_bounds)
    with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
        ocp.update_bounds(x_init, u_init)


def test_update_bounds_and_init_with_param():
    def my_parameter_function(biorbd_model, value, extra_value):
        new_gravity = MX.zeros(3, 1)
        new_gravity[2] = value + extra_value
        biorbd_model.setGravity(new_gravity)

    def my_target_function(ocp, value, target_value):
        return value + target_value

    biorbd_model = biorbd.Model(TestUtils.bioptim_folder() + "/examples/track/models/cube_and_line.bioMod")
    nq = biorbd_model.nbQ()
    ns = 10
    g_min, g_max, g_init = -10, -6, -8

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    parameters = ParameterList()
    bounds_gravity = Bounds(g_min, g_max, interpolation=InterpolationType.CONSTANT)
    initial_gravity = InitialGuess(g_init)
    parameter_objective_functions = Objective(
        my_target_function, weight=10, quadratic=True, custom_type=ObjectiveFcn.Parameter, target_value=-8
    )
    parameters.add(
        "gravity_z",
        my_parameter_function,
        initial_gravity,
        bounds_gravity,
        size=1,
        penalty_list=parameter_objective_functions,
        extra_value=1,
    )

    ocp = OptimalControlProgram(biorbd_model, dynamics, ns, 1.0, parameters=parameters)

    x_bounds = Bounds(-np.ones((nq * 2, 1)), np.ones((nq * 2, 1)))
    u_bounds = Bounds(-2.0 * np.ones((nq, 1)), 2.0 * np.ones((nq, 1)))
    ocp.update_bounds(x_bounds, u_bounds)

    expected = np.array([[-1] * (nq * 2) * (ns + 1) + [-2] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.bounds.min, np.append(expected, [g_min])[:, np.newaxis])
    expected = np.array([[1] * (nq * 2) * (ns + 1) + [2] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.bounds.max, np.append(expected, [g_max])[:, np.newaxis])

    x_init = InitialGuess(0.5 * np.ones((nq * 2, 1)))
    u_init = InitialGuess(-0.5 * np.ones((nq, 1)))
    ocp.update_initial_guess(x_init, u_init)
    expected = np.array([[0.5] * (nq * 2) * (ns + 1) + [-0.5] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.init.init, np.append(expected, [g_init])[:, np.newaxis])


def test_add_wrong_param():
    g_min, g_max, g_init = -10, -6, -8

    def my_parameter_function(biorbd_model, value, extra_value):
        biorbd_model.setGravity(biorbd.Vector3d(0, 0, value + extra_value))

    def my_target_function(ocp, value, target_value):
        return value + target_value

    parameters = ParameterList()
    initial_gravity = InitialGuess(g_init)
    bounds_gravity = Bounds(g_min, g_max, interpolation=InterpolationType.CONSTANT)
    parameter_objective_functions = Objective(
        my_target_function, weight=10, quadratic=True, custom_type=ObjectiveFcn.Parameter, target_value=-8
    )

    with pytest.raises(
        RuntimeError, match="function, initial_guess, bounds and size are mandatory elements to declare a parameter"
    ):
        parameters.add(
            "gravity_z",
            [],
            initial_gravity,
            bounds_gravity,
            size=1,
            penalty_list=parameter_objective_functions,
            extra_value=1,
        )

    with pytest.raises(
        RuntimeError, match="function, initial_guess, bounds and size are mandatory elements to declare a parameter"
    ):
        parameters.add(
            "gravity_z",
            my_parameter_function,
            None,
            bounds_gravity,
            size=1,
            penalty_list=parameter_objective_functions,
            extra_value=1,
        )

    with pytest.raises(
        RuntimeError, match="function, initial_guess, bounds and size are mandatory elements to declare a parameter"
    ):
        parameters.add(
            "gravity_z",
            my_parameter_function,
            initial_gravity,
            None,
            size=1,
            penalty_list=parameter_objective_functions,
            extra_value=1,
        )

    with pytest.raises(
        RuntimeError, match="function, initial_guess, bounds and size are mandatory elements to declare a parameter"
    ):
        parameters.add(
            "gravity_z",
            my_parameter_function,
            initial_gravity,
            bounds_gravity,
            penalty_list=parameter_objective_functions,
            extra_value=1,
        )


def test_update_noised_init():
    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model = biorbd.Model(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    ns = 5

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, ns, 1.0)

    # Path constraint and control path constraints
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * ntau, [tau_max] * ntau)

    # Initial guesses
    t = None
    x = np.zeros((nq * 2, ns + 1))
    u = np.zeros((ntau, ns))

    x_init = NoisedInitialGuess(
        x,
        t=t,
        init_interpolation=InterpolationType.EACH_FRAME,
        bounds=x_bounds,
        bound_t=None,
        noise_magnitude=0.01,
        n_elements=nq + nqdot,
        n_shooting=ns,
        bound_push=0.1,
    )
    u_init = NoisedInitialGuess(
        u,
        t=t,
        init_interpolation=InterpolationType.EACH_FRAME,
        bounds=u_bounds,
        bound_t=None,
        noise_magnitude=0.01,
        n_elements=ntau,
        n_shooting=ns - 1,
        bound_push=0.1,
    )

    ocp.update_initial_guess(x_init, u_init)
    expected = np.array(
        [
            7.89487745e-03,
            -1.00000000e-01,
            -1.00000000e-01,
            -1.00000000e-01,
            -1.00000000e-01,
            -1.00000000e-01,
            8.27317537e-03,
            8.47162999e-03,
            8.99483266e-04,
            7.66428469e-03,
            7.75000815e-05,
            2.87971956e-03,
            5.51704339e-03,
            9.93479466e-03,
            6.51457413e-04,
            1.07748305e-03,
            5.18674803e-03,
            7.67068670e-03,
            3.20049998e-03,
            4.07936811e-03,
            7.89056852e-03,
            3.88173868e-03,
            7.65550071e-03,
            8.62313189e-03,
            4.20496097e-03,
            9.86154743e-04,
            3.71147283e-03,
            9.87845348e-04,
            3.34242983e-03,
            5.04971658e-04,
            2.40754713e-03,
            -1.00000000e-01,
            1.67000000e00,
            -1.00000000e-01,
            -1.00000000e-01,
            -1.00000000e-01,
            8.69393251e-03,
            5.73129867e-03,
            9.04450191e-03,
            4.47691306e-04,
            6.76654927e-04,
            4.37545616e-03,
            5.99326365e-03,
            2.32666812e-03,
            5.21434475e-03,
            6.22547668e-03,
            1.16195380e-03,
            6.81051511e-03,
            5.68894783e-03,
            4.51552787e-03,
            2.00692106e-03,
        ]
    )

    np.testing.assert_almost_equal(ocp.v.init.init, expected[:, np.newaxis], decimal=2)
    assert ocp.v.init.init.shape == ((nq + nqdot) * (ns + 1) + ntau * ns, 1)

    with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
        ocp.update_bounds(x_init, u_init)
