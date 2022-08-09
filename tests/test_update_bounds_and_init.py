import pytest
import numpy as np
import biorbd_casadi as biorbd
from casadi import MX
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    DynamicsList,
    Bounds,
    QAndQDotBounds,
    ParameterList,
    InterpolationType,
    InitialGuess,
    NoisedInitialGuess,
    Objective,
    ObjectiveFcn,
    OdeSolver,
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
        InterpolationType.ALL_POINTS,
    ],
)
def test_update_noised_init_rk4(interpolation):
    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model = biorbd.Model(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    ns = 3
    phase_time = 1.0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(
        biorbd_model, dynamics, n_shooting=ns, phase_time=phase_time, ode_solver=OdeSolver.RK4()
    )

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
    elif interpolation == InterpolationType.ALL_POINTS:
        x = np.random.random((nq + nqdot, ns + 1))
        u = np.random.random((ntau, ns))
    elif interpolation == InterpolationType.ALL_POINTS:
        x = np.random.random((nq + nqdot, ns + 1))
        u = np.random.random((ntau, ns))
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
    if interpolation == InterpolationType.ALL_POINTS:
        with pytest.raises(ValueError, match="InterpolationType.ALL_POINTS must only be used with direct collocation"):
            ocp.update_initial_guess(x_init, u_init)
    else:
        ocp.update_initial_guess(x_init, u_init)
        print(ocp.v.init.init)

        if interpolation == InterpolationType.CONSTANT:
            expected = np.array(
                [
                    8.01464405e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    8.06455681e-01,
                    2.91788226e-03,
                    -7.32358536e-03,
                    2.67410254e-01,
                    2.08991213e-01,
                    1.87966870e-01,
                    8.03082901e-01,
                    -1.24825577e-03,
                    1.83296247e-02,
                    -2.69525994e-01,
                    1.74771041e-01,
                    -2.42032305e-02,
                    8.01346495e-01,
                    -1.00000000e-01,
                    1.47000000e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -7.63451148e-01,
                    8.89337834e-01,
                    -4.70888776e-01,
                    2.79842043e-01,
                    4.36966435e-02,
                    5.48467379e-01,
                    -7.13293425e-01,
                    -1.70676120e-01,
                    -8.76993356e-02,
                ]
            )
        elif interpolation == InterpolationType.LINEAR:
            expected = np.array(
                [
                    1.80146441e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    2.05645568e00,
                    2.91788226e-03,
                    3.85176415e-01,
                    2.67410254e-01,
                    2.08991213e-01,
                    1.87966870e-01,
                    2.20000000e00,
                    -1.24825577e-03,
                    8.03329625e-01,
                    -2.69525994e-01,
                    1.74771041e-01,
                    -2.42032305e-02,
                    2.20000000e00,
                    -1.00000000e-01,
                    1.47000000e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    6.86548852e-01,
                    1.06993378e01,
                    1.80911122e00,
                    7.63175376e-01,
                    9.85369664e00,
                    1.30846738e00,
                    -1.19662676e00,
                    9.63932388e00,
                    -8.47699336e-01,
                ]
            )
        elif interpolation == InterpolationType.SPLINE:
            expected = np.array(
                [
                    1.41356013,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.1,
                    1.56086544,
                    0.43242159,
                    0.20005561,
                    0.88294594,
                    0.55189025,
                    0.62633582,
                    1.70614069,
                    0.138338,
                    0.31053737,
                    -0.05712797,
                    0.61455967,
                    0.17849204,
                    1.37159075,
                    -0.1,
                    1.47,
                    -0.1,
                    -0.1,
                    -0.1,
                    0.05754208,
                    1.35798904,
                    -0.18808181,
                    0.80090412,
                    0.80762251,
                    0.76935078,
                    -0.50430501,
                    0.4788157,
                    0.05657921,
                ]
            )

        elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            expected = np.array(
                [
                    1.80146441e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    2.20000000e00,
                    2.91788226e-03,
                    7.77676415e-01,
                    2.67410254e-01,
                    2.08991213e-01,
                    1.87966870e-01,
                    2.20000000e00,
                    -1.24825577e-03,
                    8.03329625e-01,
                    -2.69525994e-01,
                    1.74771041e-01,
                    -2.42032305e-02,
                    2.20000000e00,
                    -1.00000000e-01,
                    1.47000000e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    6.86548852e-01,
                    1.06993378e01,
                    1.80911122e00,
                    2.79842043e-01,
                    9.85369664e00,
                    5.48467379e-01,
                    -7.13293425e-01,
                    9.63932388e00,
                    -8.76993356e-02,
                ]
            )

        if interpolation == InterpolationType.EACH_FRAME:
            expected = np.array(
                [
                    8.01464405e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    8.06455681e-01,
                    2.91788226e-03,
                    -7.32358536e-03,
                    2.67410254e-01,
                    2.08991213e-01,
                    1.87966870e-01,
                    8.03082901e-01,
                    -1.24825577e-03,
                    1.83296247e-02,
                    -2.69525994e-01,
                    1.74771041e-01,
                    -2.42032305e-02,
                    8.01346495e-01,
                    -1.00000000e-01,
                    1.47000000e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -7.63451148e-01,
                    8.89337834e-01,
                    -4.70888776e-01,
                    2.79842043e-01,
                    4.36966435e-02,
                    5.48467379e-01,
                    -7.13293425e-01,
                    -1.70676120e-01,
                    -8.76993356e-02,
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
        InterpolationType.ALL_POINTS,
    ],
)
def test_update_noised_init_collocation(interpolation):
    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model = biorbd.Model(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    ns = 3
    phase_time = 1.0
    solver = OdeSolver.COLLOCATION(polynomial_degree=1)
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, n_shooting=ns, phase_time=phase_time, ode_solver=solver)

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
        x = np.zeros((nq + nqdot, ns + 1))
        u = np.zeros((ntau, ns))
    elif interpolation == InterpolationType.ALL_POINTS:
        x = np.zeros((nq + nqdot, ns * (solver.polynomial_degree + 1) + 1))
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

    if interpolation == InterpolationType.CONSTANT:
        expected = np.array(
            [
                8.01464405e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                8.06455681e-01,
                2.91788226e-03,
                -7.32358536e-03,
                2.67410254e-01,
                2.08991213e-01,
                1.87966870e-01,
                8.06455681e-01,
                2.91788226e-03,
                -7.32358536e-03,
                2.67410254e-01,
                2.08991213e-01,
                1.87966870e-01,
                8.06455681e-01,
                2.91788226e-03,
                -7.32358536e-03,
                2.67410254e-01,
                2.08991213e-01,
                1.87966870e-01,
                8.03082901e-01,
                -1.24825577e-03,
                1.83296247e-02,
                -2.69525994e-01,
                1.74771041e-01,
                -2.42032305e-02,
                8.03082901e-01,
                -1.24825577e-03,
                1.83296247e-02,
                -2.69525994e-01,
                1.74771041e-01,
                -2.42032305e-02,
                8.01346495e-01,
                -1.00000000e-01,
                1.47000000e+00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -7.63451148e-01,
                8.89337834e-01,
                -4.70888776e-01,
                2.79842043e-01,
                4.36966435e-02,
                5.48467379e-01,
                -7.13293425e-01,
                -1.70676120e-01,
                -8.76993356e-02,
            ]
        )
    elif interpolation == InterpolationType.LINEAR:
        expected = np.array(
            [
                1.80146441e+00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                2.05645568e+00,
                2.91788226e-03,
                3.85176415e-01,
                2.67410254e-01,
                2.08991213e-01,
                1.87966870e-01,
                2.05645568e+00,
                2.91788226e-03,
                3.85176415e-01,
                2.67410254e-01,
                2.08991213e-01,
                1.87966870e-01,
                2.05645568e+00,
                2.91788226e-03,
                3.85176415e-01,
                2.67410254e-01,
                2.08991213e-01,
                1.87966870e-01,
                2.20000000e+00,
                -1.24825577e-03,
                8.03329625e-01,
                -2.69525994e-01,
                1.74771041e-01,
                -2.42032305e-02,
                2.20000000e+00,
                -1.24825577e-03,
                8.03329625e-01,
                -2.69525994e-01,
                1.74771041e-01,
                -2.42032305e-02,
                2.20000000e+00,
                -1.00000000e-01,
                1.47000000e+00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                6.86548852e-01,
                1.06993378e+01,
                1.80911122e+00,
                7.63175376e-01,
                9.85369664e+00,
                1.30846738e+00,
                -1.19662676e+00,
                9.63932388e+00,
                -8.47699336e-01,
            ]
        )
    elif interpolation == InterpolationType.SPLINE:
        expected = np.array(
            [
                1.41356013,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                1.56086544,
                0.43242159,
                0.20005561,
                0.88294594,
                0.55189025,
                0.62633582,
                1.56086544,
                0.43242159,
                0.20005561,
                0.88294594,
                0.55189025,
                0.62633582,
                1.56086544,
                0.43242159,
                0.20005561,
                0.88294594,
                0.55189025,
                0.62633582,
                1.70614069,
                0.138338,
                0.31053737,
                -0.05712797,
                0.61455967,
                0.17849204,
                1.70614069,
                0.138338,
                0.31053737,
                -0.05712797,
                0.61455967,
                0.17849204,
                1.37159075,
                -0.1,
                1.47,
                -0.1,
                -0.1,
                -0.1,
                0.05754208,
                1.35798904,
                -0.18808181,
                0.80090412,
                0.80762251,
                0.76935078,
                -0.50430501,
                0.4788157,
                0.05657921,
            ]
        )

    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        expected = np.array(
            [
                1.80146441e+00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                2.20000000e+00,
                2.91788226e-03,
                7.77676415e-01,
                2.67410254e-01,
                2.08991213e-01,
                1.87966870e-01,
                2.20000000e+00,
                2.91788226e-03,
                7.77676415e-01,
                2.67410254e-01,
                2.08991213e-01,
                1.87966870e-01,
                2.20000000e+00,
                2.91788226e-03,
                7.77676415e-01,
                2.67410254e-01,
                2.08991213e-01,
                1.87966870e-01,
                2.20000000e+00,
                -1.24825577e-03,
                8.03329625e-01,
                -2.69525994e-01,
                1.74771041e-01,
                -2.42032305e-02,
                2.20000000e+00,
                -1.24825577e-03,
                8.03329625e-01,
                -2.69525994e-01,
                1.74771041e-01,
                -2.42032305e-02,
                2.20000000e+00,
                -1.00000000e-01,
                1.47000000e+00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                6.86548852e-01,
                1.06993378e+01,
                1.80911122e+00,
                2.79842043e-01,
                9.85369664e+00,
                5.48467379e-01,
                -7.13293425e-01,
                9.63932388e+00,
                -8.76993356e-02,
            ]
        )

    elif interpolation == InterpolationType.EACH_FRAME:
        expected = np.array(
            [
                8.01464405e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                8.06455681e-01,
                2.91788226e-03,
                -7.32358536e-03,
                2.67410254e-01,
                2.08991213e-01,
                1.87966870e-01,
                8.06455681e-01,
                2.91788226e-03,
                -7.32358536e-03,
                2.67410254e-01,
                2.08991213e-01,
                1.87966870e-01,
                8.06455681e-01,
                2.91788226e-03,
                -7.32358536e-03,
                2.67410254e-01,
                2.08991213e-01,
                1.87966870e-01,
                8.03082901e-01,
                -1.24825577e-03,
                1.83296247e-02,
                -2.69525994e-01,
                1.74771041e-01,
                -2.42032305e-02,
                8.03082901e-01,
                -1.24825577e-03,
                1.83296247e-02,
                -2.69525994e-01,
                1.74771041e-01,
                -2.42032305e-02,
                8.01346495e-01,
                -1.00000000e-01,
                1.47000000e+00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -7.63451148e-01,
                8.89337834e-01,
                -4.70888776e-01,
                2.79842043e-01,
                4.36966435e-02,
                5.48467379e-01,
                -7.13293425e-01,
                -1.70676120e-01,
                -8.76993356e-02,
            ]
        )

    elif interpolation == InterpolationType.ALL_POINTS:
        expected = np.array(
            [
                8.01464405e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                8.06455681e-01,
                9.27325521e-03,
                -2.59414312e-02,
                -2.42032305e-02,
                -5.36194845e-02,
                7.04318198e-02,
                8.03082901e-01,
                -2.33116962e-03,
                -3.01455672e-02,
                1.76261680e-01,
                -1.47934072e-01,
                7.34717971e-02,
                8.01346495e-01,
                5.83450076e-03,
                2.08991213e-02,
                -2.39845252e-01,
                1.72306109e-01,
                2.78815141e-01,
                7.97709644e-01,
                5.77898395e-04,
                1.74771041e-02,
                8.79149705e-02,
                -2.75515588e-02,
                1.14241063e-01,
                8.04376823e-01,
                1.36089122e-03,
                2.32485489e-02,
                -2.24087738e-01,
                4.29983182e-02,
                -8.82737895e-02,
                7.98127616e-01,
                -1.00000000e-01,
                1.47000000e+00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                3.95262392e-01,
                3.41275739e-01,
                -3.69143298e-01,
                -8.79549057e-01,
                -5.79234878e-01,
                -2.72578458e-01,
                3.33533431e-01,
                -7.42147405e-01,
                1.40393541e-01,
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
        InterpolationType.ALL_POINTS,
    ],
)
def test_update_noised_initial_guess_rk4(interpolation):
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
    elif interpolation == InterpolationType.ALL_POINTS:
        x = InitialGuess(np.zeros((nq + nqdot, ns + 1)), interpolation=interpolation)
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
    if interpolation == InterpolationType.ALL_POINTS:
        with pytest.raises(ValueError, match="InterpolationType.ALL_POINTS must only be used with direct collocation"):
            ocp.update_initial_guess(x_init, u_init)
    else:
        ocp.update_initial_guess(x_init, u_init)
        print(ocp.v.init.init)
        if interpolation == InterpolationType.CONSTANT:
            expected = np.array(
                [
                    0.7962362,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.1,
                    0.81352143,
                    -0.00688011,
                    0.01307359,
                    -0.18074267,
                    0.01555492,
                    -0.22651269,
                    0.80695982,
                    -0.00883833,
                    -0.03012256,
                    -0.19991527,
                    -0.04276021,
                    -0.13059937,
                    0.80295975,
                    -0.1,
                    1.47,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.25091976,
                    0.19731697,
                    -0.88383278,
                    0.90142861,
                    -0.68796272,
                    0.73235229,
                    0.46398788,
                    -0.68801096,
                    0.20223002,
                ]
            )
        elif interpolation == InterpolationType.LINEAR:
            expected = np.array(
                [
                    1.79623620e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    2.06352143e00,
                    -6.88010959e-03,
                    4.05573586e-01,
                    -1.80742667e-01,
                    1.55549247e-02,
                    -2.26512688e-01,
                    2.20000000e00,
                    -8.83832776e-03,
                    7.54877435e-01,
                    -1.99915269e-01,
                    -4.27602059e-02,
                    -1.30599369e-01,
                    2.20000000e00,
                    -1.00000000e-01,
                    1.47000000e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    1.19908024e00,
                    1.00073170e01,
                    1.39616722e00,
                    1.38476195e00,
                    9.12203728e00,
                    1.49235229e00,
                    -1.93454497e-02,
                    9.12198904e00,
                    -5.57769977e-01,
                ]
            )
        elif interpolation == InterpolationType.SPLINE:
            expected = np.array(
                [
                    1.39489469e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    1.11672435e00,
                    6.65627498e-01,
                    2.05044956e-01,
                    1.57276580e-01,
                    4.41795628e-01,
                    1.47886691e-03,
                    9.62969993e-01,
                    4.57938262e-01,
                    1.52256794e-01,
                    2.03847059e-01,
                    5.28820074e-01,
                    1.12785129e-01,
                    9.50893802e-01,
                    -1.00000000e-01,
                    1.47000000e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    6.97965775e-01,
                    8.81549995e-01,
                    2.54876264e-02,
                    1.86521820e00,
                    -2.20956562e-01,
                    1.06270452e00,
                    1.30112101e00,
                    -5.07835040e-01,
                    7.90965478e-01,
                ]
            )

        elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            expected = np.array(
                [
                    1.79623620e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    2.20000000e00,
                    -6.88010959e-03,
                    7.98073586e-01,
                    -1.80742667e-01,
                    1.55549247e-02,
                    -2.26512688e-01,
                    2.20000000e00,
                    -8.83832776e-03,
                    7.54877435e-01,
                    -1.99915269e-01,
                    -4.27602059e-02,
                    -1.30599369e-01,
                    2.20000000e00,
                    -1.00000000e-01,
                    1.47000000e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    1.19908024e00,
                    1.00073170e01,
                    1.39616722e00,
                    9.01428613e-01,
                    9.12203728e00,
                    7.32352292e-01,
                    4.63987884e-01,
                    9.12198904e00,
                    2.02230023e-01,
                ]
            )

        if interpolation == InterpolationType.EACH_FRAME:
            expected = np.array(
                [
                    0.7962362,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.1,
                    0.81352143,
                    -0.00688011,
                    0.01307359,
                    -0.18074267,
                    0.01555492,
                    -0.22651269,
                    0.80695982,
                    -0.00883833,
                    -0.03012256,
                    -0.19991527,
                    -0.04276021,
                    -0.13059937,
                    0.80295975,
                    -0.1,
                    1.47,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.25091976,
                    0.19731697,
                    -0.88383278,
                    0.90142861,
                    -0.68796272,
                    0.73235229,
                    0.46398788,
                    -0.68801096,
                    0.20223002,
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
        InterpolationType.ALL_POINTS,
    ],
)
def test_update_noised_initial_guess_collocation(interpolation):
    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model = biorbd.Model(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    ns = 3
    phase_time = 1.0
    solver = OdeSolver.COLLOCATION(polynomial_degree=1)

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, n_shooting=ns, phase_time=phase_time, ode_solver=solver)

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
    elif interpolation == InterpolationType.ALL_POINTS:
        x = InitialGuess(np.zeros((nq + nqdot, ns * (solver.polynomial_degree + 1) + 1)), interpolation=interpolation)
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
    if interpolation == InterpolationType.CONSTANT:
        expected = np.array(
            [
                0.7962362,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                0.81352143,
                -0.00688011,
                0.01307359,
                -0.18074267,
                0.01555492,
                -0.22651269,
                0.81352143,
                -0.00688011,
                0.01307359,
                -0.18074267,
                0.01555492,
                -0.22651269,
                0.81352143,
                -0.00688011,
                0.01307359,
                -0.18074267,
                0.01555492,
                -0.22651269,
                0.80695982,
                -0.00883833,
                -0.03012256,
                -0.19991527,
                -0.04276021,
                -0.13059937,
                0.80695982,
                -0.00883833,
                -0.03012256,
                -0.19991527,
                -0.04276021,
                -0.13059937,
                0.80295975,
                -0.1,
                1.47,
                -0.1,
                -0.1,
                -0.1,
                -0.25091976,
                0.19731697,
                -0.88383278,
                0.90142861,
                -0.68796272,
                0.73235229,
                0.46398788,
                -0.68801096,
                0.20223002,
            ]
        )
    elif interpolation == InterpolationType.LINEAR:
        expected = np.array(
            [
                1.79623620e+00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                2.06352143e+00,
                -6.88010959e-03,
                4.05573586e-01,
                -1.80742667e-01,
                1.55549247e-02,
                -2.26512688e-01,
                2.06352143e+00,
                -6.88010959e-03,
                4.05573586e-01,
                -1.80742667e-01,
                1.55549247e-02,
                -2.26512688e-01,
                2.06352143e+00,
                -6.88010959e-03,
                4.05573586e-01,
                -1.80742667e-01,
                1.55549247e-02,
                -2.26512688e-01,
                2.20000000e+00,
                -8.83832776e-03,
                7.54877435e-01,
                -1.99915269e-01,
                -4.27602059e-02,
                -1.30599369e-01,
                2.20000000e+00,
                -8.83832776e-03,
                7.54877435e-01,
                -1.99915269e-01,
                -4.27602059e-02,
                -1.30599369e-01,
                2.20000000e+00,
                -1.00000000e-01,
                1.47000000e+00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                1.19908024e+00,
                1.00073170e+01,
                1.39616722e+00,
                1.38476195e+00,
                9.12203728e+00,
                1.49235229e+00,
                -1.93454497e-02,
                9.12198904e+00,
                -5.57769977e-01,
            ]
        )
    elif interpolation == InterpolationType.SPLINE:
        expected = np.array(
            [
                1.39489469e+00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                1.11672435e+00,
                6.65627498e-01,
                2.05044956e-01,
                1.57276580e-01,
                4.41795628e-01,
                1.47886691e-03,
                1.11672435e+00,
                6.65627498e-01,
                2.05044956e-01,
                1.57276580e-01,
                4.41795628e-01,
                1.47886691e-03,
                1.11672435e+00,
                6.65627498e-01,
                2.05044956e-01,
                1.57276580e-01,
                4.41795628e-01,
                1.47886691e-03,
                9.62969993e-01,
                4.57938262e-01,
                1.52256794e-01,
                2.03847059e-01,
                5.28820074e-01,
                1.12785129e-01,
                9.62969993e-01,
                4.57938262e-01,
                1.52256794e-01,
                2.03847059e-01,
                5.28820074e-01,
                1.12785129e-01,
                9.50893802e-01,
                -1.00000000e-01,
                1.47000000e+00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                6.97965775e-01,
                8.81549995e-01,
                2.54876264e-02,
                1.86521820e+00,
                -2.20956562e-01,
                1.06270452e+00,
                1.30112101e+00,
                -5.07835040e-01,
                7.90965478e-01,
            ]
        )
    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        expected = np.array(
            [
                1.79623620e+00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                2.20000000e+00,
                -6.88010959e-03,
                7.98073586e-01,
                -1.80742667e-01,
                1.55549247e-02,
                -2.26512688e-01,
                2.20000000e+00,
                -6.88010959e-03,
                7.98073586e-01,
                -1.80742667e-01,
                1.55549247e-02,
                -2.26512688e-01,
                2.20000000e+00,
                -6.88010959e-03,
                7.98073586e-01,
                -1.80742667e-01,
                1.55549247e-02,
                -2.26512688e-01,
                2.20000000e+00,
                -8.83832776e-03,
                7.54877435e-01,
                -1.99915269e-01,
                -4.27602059e-02,
                -1.30599369e-01,
                2.20000000e+00,
                -8.83832776e-03,
                7.54877435e-01,
                -1.99915269e-01,
                -4.27602059e-02,
                -1.30599369e-01,
                2.20000000e+00,
                -1.00000000e-01,
                1.47000000e+00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                1.19908024e+00,
                1.00073170e+01,
                1.39616722e+00,
                9.01428613e-01,
                9.12203728e+00,
                7.32352292e-01,
                4.63987884e-01,
                9.12198904e+00,
                2.02230023e-01,
            ]
        )
    elif interpolation == InterpolationType.EACH_FRAME:
        expected = np.array(
            [
                0.7962362,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                0.81352143,
                -0.00688011,
                0.01307359,
                -0.18074267,
                0.01555492,
                -0.22651269,
                0.81352143,
                -0.00688011,
                0.01307359,
                -0.18074267,
                0.01555492,
                -0.22651269,
                0.81352143,
                -0.00688011,
                0.01307359,
                -0.18074267,
                0.01555492,
                -0.22651269,
                0.80695982,
                -0.00883833,
                -0.03012256,
                -0.19991527,
                -0.04276021,
                -0.13059937,
                0.80695982,
                -0.00883833,
                -0.03012256,
                -0.19991527,
                -0.04276021,
                -0.13059937,
                0.80295975,
                -0.1,
                1.47,
                -0.1,
                -0.1,
                -0.1,
                -0.25091976,
                0.19731697,
                -0.88383278,
                0.90142861,
                -0.68796272,
                0.73235229,
                0.46398788,
                -0.68801096,
                0.20223002,
            ]
        )
    elif interpolation == InterpolationType.ALL_POINTS:
        expected = np.array(
            [
                0.7962362,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                0.81352143,
                0.0020223,
                -0.01989228,
                -0.13059937,
                -0.28497361,
                -0.12276479,
                0.80695982,
                0.00416145,
                -0.01229982,
                -0.08396733,
                0.06757242,
                -0.25279007,
                0.80295975,
                -0.00958831,
                0.00155549,
                -0.02760204,
                -0.2070158,
                0.11575702,
                0.78968056,
                0.0093982,
                -0.00427602,
                0.17918134,
                -0.27328614,
                -0.0376033,
                0.78967984,
                0.00664885,
                -0.01311746,
                -0.18870053,
                0.2820431,
                -0.23748038,
                0.78674251,
                -0.1,
                1.47,
                -0.1,
                -0.1,
                -0.1,
                -0.25091976,
                0.19731697,
                -0.88383278,
                0.90142861,
                -0.68796272,
                0.73235229,
                0.46398788,
                -0.68801096,
                0.20223002,
            ]
        )

    np.testing.assert_almost_equal(ocp.v.init.init, expected[:, np.newaxis])
    with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
        ocp.update_bounds(x_init, u_init)
