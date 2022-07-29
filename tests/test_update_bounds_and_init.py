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


@pytest.mark.parametrize(
    "interpolation",
    [
        InterpolationType.CONSTANT,
        InterpolationType.LINEAR,
        InterpolationType.SPLINE,
        InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        InterpolationType.EACH_FRAME,
    ],
)
def test_update_noised_init(interpolation):
    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model = biorbd.Model(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    ns = 3
    phase_time = 1.0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, n_shooting=ns, phase_time=phase_time)

    # Path constraint and control path constraints
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * ntau, [tau_max] * ntau)

    # Initial guesses
    t = None
    extra_params_x = {}
    extra_params_u = {}
    if interpolation == InterpolationType.CONSTANT:
        x = [0] * (nq + nqdot)
        u = [tau_init] * ntau
    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        x = np.array([[1.0, 0.0, 0.0, 0, 0, 0], [1.5, 0.0, 0.785, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T
        u = np.array([[1.45, 9.81, 2.28], [0, 9.81, 0], [-1.45, 9.81, -2.28]]).T
    elif interpolation == InterpolationType.LINEAR:
        x = np.array([[1.0, 0.0, 0.0, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T
        u = np.array([[1.45, 9.81, 2.28], [-1.45, 9.81, -2.28]]).T
    elif interpolation == InterpolationType.EACH_FRAME:
        x = np.zeros((nq * 2, ns + 1))
        u = np.zeros((ntau, ns))
    elif interpolation == InterpolationType.SPLINE:
        # Bound , assume the first and last point are 0 and final respectively
        t = np.hstack((0, np.sort(np.random.random((3,)) * phase_time), phase_time))
        x = np.random.random((nq + nqdot, 5))
        u = np.random.random((ntau, 5))

    np.random.seed(0)
    x_init = NoisedInitialGuess(
        initial_guess=x,
        t=t,
        interpolation=interpolation,
        bounds=x_bounds,
        noise_magnitude=0.01,
        n_shooting=ns,
        bound_push=0.1,
        **extra_params_x,
    )
    u_init = NoisedInitialGuess(
        u,
        t=t,
        interpolation=interpolation,
        bounds=u_bounds,
        noise_magnitude=0.01,
        n_shooting=ns - 1,
        bound_push=0.1,
        **extra_params_u,
    )

    ocp.update_initial_guess(x_init, u_init)
    print(ocp.v.init.init)
    if interpolation == InterpolationType.EACH_FRAME:
        expected = np.array(
            [
                -0.6,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.6,
                -0.9,
                -3.04159265,
                -30.83435702,
                -30.89277606,
                -30.9138004,
                -0.6,
                -0.9,
                -3.04159265,
                -31.31592654,
                -30.92699623,
                -31.1259705,
                -0.6,
                -0.1,
                1.47,
                -0.1,
                -0.1,
                -0.1,
                -99.76345115,
                -98.11066217,
                -99.47088878,
                -98.72015796,
                -98.95630336,
                -98.45153262,
                -99.71329343,
                -99.17067612,
                -99.08769934,
            ]
        )
    elif interpolation == InterpolationType.SPLINE:
        expected = np.array(
            [
                -7.14398722e-02,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                7.58654415e-02,
                -5.57578410e-01,
                -2.91012112e00,
                -3.02188213e01,
                -3.05498770e01,
                -3.04754314e01,
                2.21140693e-01,
                -8.51662000e-01,
                -2.79963936e00,
                -3.11588952e01,
                -3.04872076e01,
                -3.09232752e01,
                -1.13409254e-01,
                -1.00000000e-01,
                1.47000000e00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -9.89424579e01,
                -9.76420110e01,
                -9.91880818e01,
                -9.81990959e01,
                -9.81923775e01,
                -9.82306492e01,
                -9.95043050e01,
                -9.85211843e01,
                -9.89434208e01,
            ]
        )
    elif interpolation == InterpolationType.LINEAR:
        expected = np.array(
            [
                0.31646441,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                0.57145568,
                -0.9,
                -2.72500031,
                -30.83435702,
                -30.89277606,
                -30.9138004,
                0.8180829,
                -0.9,
                -2.3068471,
                -31.31592654,
                -30.92699623,
                -31.1259705,
                1.0663465,
                -0.1,
                1.47,
                -0.1,
                -0.1,
                -0.1,
                -98.31345115,
                -88.30066217,
                -97.19088878,
                -98.23682462,
                -89.14630336,
                -97.69153262,
                -99.9,
                -89.36067612,
                -99.84769934,
            ]
        )
    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        expected = np.array(
            [
                0.31646441,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                0.82145568,
                -0.9,
                -2.33250031,
                -30.83435702,
                -30.89277606,
                -30.9138004,
                0.8180829,
                -0.9,
                -2.3068471,
                -31.31592654,
                -30.92699623,
                -31.1259705,
                0.8163465,
                -0.1,
                1.47,
                -0.1,
                -0.1,
                -0.1,
                -98.31345115,
                -88.30066217,
                -97.19088878,
                -98.72015796,
                -89.14630336,
                -98.45153262,
                -99.71329343,
                -89.36067612,
                -99.08769934,
            ]
        )
    elif interpolation == InterpolationType.CONSTANT:
        expected = np.array(
            [
                -0.6,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.6,
                -0.9,
                -3.04159265,
                -30.83435702,
                -30.89277606,
                -30.9138004,
                -0.6,
                -0.9,
                -3.04159265,
                -31.31592654,
                -30.92699623,
                -31.1259705,
                -0.6,
                -0.1,
                1.47,
                -0.1,
                -0.1,
                -0.1,
                -99.76345115,
                -98.11066217,
                -99.47088878,
                -98.72015796,
                -98.95630336,
                -98.45153262,
                -99.71329343,
                -99.17067612,
                -99.08769934,
            ]
        )

    np.testing.assert_almost_equal(ocp.v.init.init, expected[:, np.newaxis])

    with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
        ocp.update_bounds(x_init, u_init)


@pytest.mark.parametrize(
    "interpolation",
    [
        InterpolationType.CONSTANT,
        InterpolationType.LINEAR,
        InterpolationType.SPLINE,
        InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        InterpolationType.EACH_FRAME,
    ],
)
def test_update_noised_initial_guess(interpolation):
    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model = biorbd.Model(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    ns = 3
    phase_time = 1.0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, n_shooting=ns, phase_time=phase_time)

    # Path constraint and control path constraints
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * ntau, [tau_max] * ntau)

    # Initial guesses
    t = None
    extra_params_x = {}
    extra_params_u = {}
    if interpolation == InterpolationType.CONSTANT:
        x = InitialGuess([0] * (nq + nqdot), interpolation=interpolation)
        u = InitialGuess([tau_init] * ntau, interpolation=interpolation)
    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        x = InitialGuess(
            np.array([[1.0, 0.0, 0.0, 0, 0, 0], [1.5, 0.0, 0.785, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T,
            interpolation=interpolation,
        )
        u = InitialGuess(
            np.array([[1.45, 9.81, 2.28], [0, 9.81, 0], [-1.45, 9.81, -2.28]]).T, interpolation=interpolation
        )
    elif interpolation == InterpolationType.LINEAR:
        x = InitialGuess(np.array([[1.0, 0.0, 0.0, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T, interpolation=interpolation)
        u = InitialGuess(np.array([[1.45, 9.81, 2.28], [-1.45, 9.81, -2.28]]).T, interpolation=interpolation)
    elif interpolation == InterpolationType.EACH_FRAME:
        x = InitialGuess(np.zeros((nq * 2, ns + 1)), interpolation=interpolation)
        u = InitialGuess(np.zeros((ntau, ns)), interpolation=interpolation)
    elif interpolation == InterpolationType.SPLINE:
        # Bound spline assume the first and last point are 0 and final respectively
        np.random.seed(42)
        t = np.hstack((0, np.sort(np.random.random((3,)) * phase_time), phase_time))
        x = InitialGuess(np.random.random((nq + nqdot, 5)), interpolation=interpolation, t=t)
        u = InitialGuess(np.random.random((ntau, 5)), interpolation=interpolation, t=t)

    x_init = NoisedInitialGuess(
        initial_guess=x,
        bounds=x_bounds,
        noise_magnitude=0.01,
        n_shooting=ns,
        bound_push=0.1,
        seed=42,
        **extra_params_x,
    )
    u_init = NoisedInitialGuess(
        initial_guess=u,
        bounds=u_bounds,
        noise_magnitude=0.01,
        n_shooting=ns - 1,
        bound_push=0.1,
        seed=42,
        **extra_params_u,
    )

    ocp.update_initial_guess(x_init, u_init)
    print(ocp.v.init.init)
    if interpolation == InterpolationType.EACH_FRAME:
        expected = np.array(
            [
                -0.6,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.6,
                -0.9,
                -3.04159265,
                -31.28250994,
                -31.08621235,
                -31.31592654,
                -0.6,
                -0.9,
                -3.04159265,
                -31.30168254,
                -31.14452748,
                -31.23236664,
                -0.6,
                -0.1,
                1.47,
                -0.1,
                -0.1,
                -0.1,
                -99.25091976,
                -98.80268303,
                -99.88383278,
                -98.09857139,
                -99.68796272,
                -98.26764771,
                -98.53601212,
                -99.68801096,
                -98.79776998,
            ]
        )
    elif interpolation == InterpolationType.SPLINE:
        expected = np.array(
            [
                -9.01053122e-02,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -3.68275648e-01,
                -3.24372502e-01,
                -2.90513177e00,
                -3.09444907e01,
                -3.06599716e01,
                -3.11002884e01,
                -5.22030007e-01,
                -5.32061738e-01,
                -2.95791993e00,
                -3.08979202e01,
                -3.05729472e01,
                -3.09889821e01,
                -5.34106198e-01,
                -1.00000000e-01,
                1.47000000e00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -9.83020342e01,
                -9.81184500e01,
                -9.89745124e01,
                -9.71347818e01,
                -9.92209566e01,
                -9.79372955e01,
                -9.76988790e01,
                -9.95078350e01,
                -9.82090345e01,
            ]
        )
    elif interpolation == InterpolationType.LINEAR:
        expected = np.array(
            [
                0.3112362,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                0.57852143,
                -0.9,
                -2.70460314,
                -31.28250994,
                -31.08621235,
                -31.31592654,
                0.82195982,
                -0.9,
                -2.35529929,
                -31.30168254,
                -31.14452748,
                -31.23236664,
                1.06795975,
                -0.1,
                1.47,
                -0.1,
                -0.1,
                -0.1,
                -97.80091976,
                -88.99268303,
                -97.60383278,
                -97.61523805,
                -89.87796272,
                -97.50764771,
                -99.01934545,
                -89.87801096,
                -99.55776998,
            ]
        )
    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        expected = np.array(
            [
                0.3112362,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                0.82852143,
                -0.9,
                -2.31210314,
                -31.28250994,
                -31.08621235,
                -31.31592654,
                0.82195982,
                -0.9,
                -2.35529929,
                -31.30168254,
                -31.14452748,
                -31.23236664,
                0.81795975,
                -0.1,
                1.47,
                -0.1,
                -0.1,
                -0.1,
                -97.80091976,
                -88.99268303,
                -97.60383278,
                -98.09857139,
                -89.87796272,
                -98.26764771,
                -98.53601212,
                -89.87801096,
                -98.79776998,
            ]
        )
    elif interpolation == InterpolationType.CONSTANT:
        expected = np.array(
            [
                -0.6,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.6,
                -0.9,
                -3.04159265,
                -31.28250994,
                -31.08621235,
                -31.31592654,
                -0.6,
                -0.9,
                -3.04159265,
                -31.30168254,
                -31.14452748,
                -31.23236664,
                -0.6,
                -0.1,
                1.47,
                -0.1,
                -0.1,
                -0.1,
                -99.25091976,
                -98.80268303,
                -99.88383278,
                -98.09857139,
                -99.68796272,
                -98.26764771,
                -98.53601212,
                -99.68801096,
                -98.79776998,
            ]
        )

    np.testing.assert_almost_equal(ocp.v.init.init, expected[:, np.newaxis])

    with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
        ocp.update_bounds(x_init, u_init)
