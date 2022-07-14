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
        # Bound spline assume the first and last point are 0 and final respectively
        t = np.hstack((0, np.sort(np.random.random((3,)) * phase_time), phase_time))
        x = np.random.random((nq + nqdot, 5))
        u = np.random.random((ntau, 5))
    elif interpolation == InterpolationType.CUSTOM:
        # The custom function refers to the one at the beginning of the file. It emulates a Linear interpolation
        from bioptim.examples.getting_started import custom_initial_guess as ocp_module

        x = ocp_module.custom_init_func
        u = ocp_module.custom_init_func
        extra_params_x = {"my_values": np.random.random((nq + nqdot, 2)), "n_shooting_custom": ns}
        extra_params_u = {"my_values": np.random.random((ntau, 2)), "n_shooting_custom": ns}

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
                5.48813504e-03,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                7.15189366e-03,
                6.45894113e-03,
                3.83441519e-03,
                9.25596638e-03,
                8.32619846e-03,
                7.99158564e-03,
                6.02763376e-03,
                4.37587211e-03,
                7.91725038e-03,
                7.10360582e-04,
                7.78156751e-03,
                4.61479362e-03,
                5.44883183e-03,
                -1.00000000e-01,
                1.67000000e00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                1.18274426e-03,
                9.44668917e-03,
                2.64555612e-03,
                6.39921021e-03,
                5.21848322e-03,
                7.74233689e-03,
                1.43353287e-03,
                4.14661940e-03,
                4.56150332e-03,
            ]
        )
    elif interpolation == InterpolationType.SPLINE:
        expected = np.array(
            [
                0.61758386,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                0.76156165,
                0.43596265,
                0.21121361,
                0.62479165,
                0.35122524,
                0.44636054,
                0.90908543,
                0.14396213,
                0.30012499,
                0.21310838,
                0.44757019,
                0.20731007,
                0.57569308,
                -0.1,
                1.67,
                -0.1,
                -0.1,
                -0.1,
                0.82217597,
                0.47809789,
                0.28545252,
                0.52746129,
                0.76914434,
                0.22862574,
                0.21042195,
                0.65363844,
                0.14884005,
            ]
        )
    elif interpolation == InterpolationType.LINEAR:
        expected = np.array(
            [
                1.00548814e00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                1.25715189e00,
                6.45894113e-03,
                3.96334415e-01,
                9.25596638e-03,
                8.32619846e-03,
                7.99158564e-03,
                1.50602763e00,
                4.37587211e-03,
                7.92917250e-01,
                7.10360582e-04,
                7.78156751e-03,
                4.61479362e-03,
                1.75544883e00,
                -1.00000000e-01,
                1.67000000e00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                1.45118274e00,
                9.81944669e00,
                2.28264556e00,
                4.89732544e-01,
                9.81521848e00,
                7.67742337e-01,
                -4.81899800e-01,
                9.81414662e00,
                -7.55438497e-01,
            ]
        )
    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        expected = np.array(
            [
                1.00548814e00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                1.50715189e00,
                6.45894113e-03,
                7.88834415e-01,
                9.25596638e-03,
                8.32619846e-03,
                7.99158564e-03,
                1.50602763e00,
                4.37587211e-03,
                7.92917250e-01,
                7.10360582e-04,
                7.78156751e-03,
                4.61479362e-03,
                1.50544883e00,
                -1.00000000e-01,
                1.67000000e00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                1.45118274e00,
                9.81944669e00,
                2.28264556e00,
                6.39921021e-03,
                9.81521848e00,
                7.74233689e-03,
                1.43353287e-03,
                9.81414662e00,
                4.56150332e-03,
            ]
        )
    elif interpolation == InterpolationType.CONSTANT:
        expected = np.array(
            [
                5.48813504e-03,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                7.15189366e-03,
                6.45894113e-03,
                3.83441519e-03,
                9.25596638e-03,
                8.32619846e-03,
                7.99158564e-03,
                6.02763376e-03,
                4.37587211e-03,
                7.91725038e-03,
                7.10360582e-04,
                7.78156751e-03,
                4.61479362e-03,
                5.44883183e-03,
                -1.00000000e-01,
                1.67000000e00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                1.18274426e-03,
                9.44668917e-03,
                2.64555612e-03,
                6.39921021e-03,
                5.21848322e-03,
                7.74233689e-03,
                1.43353287e-03,
                4.14661940e-03,
                4.56150332e-03,
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
                3.74540119e-03,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                9.50714306e-03,
                1.55994520e-03,
                7.08072578e-03,
                2.12339111e-03,
                5.24756432e-03,
                1.39493861e-03,
                7.31993942e-03,
                5.80836122e-04,
                2.05844943e-04,
                1.81824967e-03,
                4.31945019e-03,
                2.92144649e-03,
                5.98658484e-03,
                -1.00000000e-01,
                1.67000000e00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                3.74540119e-03,
                5.98658484e-03,
                5.80836122e-04,
                9.50714306e-03,
                1.56018640e-03,
                8.66176146e-03,
                7.31993942e-03,
                1.55994520e-03,
                6.01115012e-03,
            ]
        )
    elif interpolation == InterpolationType.SPLINE:
        expected = np.array(
            [
                0.70716515,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                0.27280629,
                0.68827069,
                0.49877703,
                0.48777798,
                0.36165296,
                0.71752962,
                0.07549563,
                0.55912336,
                0.52387268,
                0.41863,
                0.46839814,
                0.85563485,
                0.13823782,
                -0.1,
                1.67,
                -0.1,
                -0.1,
                -0.1,
                0.86912259,
                0.58175155,
                0.13187976,
                0.64817702,
                0.77435203,
                0.32653823,
                0.54295814,
                0.4387764,
                0.43907385,
            ]
        )
    elif interpolation == InterpolationType.LINEAR:
        expected = np.array(
            [
                1.00374540e00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                1.25950714e00,
                1.55994520e-03,
                3.99580726e-01,
                2.12339111e-03,
                5.24756432e-03,
                1.39493861e-03,
                1.50731994e00,
                5.80836122e-04,
                7.85205845e-01,
                1.81824967e-03,
                4.31945019e-03,
                2.92144649e-03,
                1.75598658e00,
                -1.00000000e-01,
                1.67000000e00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                1.45374540e00,
                9.81598658e00,
                2.28058084e00,
                4.92840476e-01,
                9.81156019e00,
                7.68661761e-01,
                -4.76013394e-01,
                9.81155995e00,
                -7.53988850e-01,
            ]
        )
    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        expected = np.array(
            [
                1.00374540e00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                1.50950714e00,
                1.55994520e-03,
                7.92080726e-01,
                2.12339111e-03,
                5.24756432e-03,
                1.39493861e-03,
                1.50731994e00,
                5.80836122e-04,
                7.85205845e-01,
                1.81824967e-03,
                4.31945019e-03,
                2.92144649e-03,
                1.50598658e00,
                -1.00000000e-01,
                1.67000000e00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                1.45374540e00,
                9.81598658e00,
                2.28058084e00,
                9.50714306e-03,
                9.81156019e00,
                8.66176146e-03,
                7.31993942e-03,
                9.81155995e00,
                6.01115012e-03,
            ]
        )
    elif interpolation == InterpolationType.CONSTANT:
        expected = np.array(
            [
                3.74540119e-03,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                9.50714306e-03,
                1.55994520e-03,
                7.08072578e-03,
                2.12339111e-03,
                5.24756432e-03,
                1.39493861e-03,
                7.31993942e-03,
                5.80836122e-04,
                2.05844943e-04,
                1.81824967e-03,
                4.31945019e-03,
                2.92144649e-03,
                5.98658484e-03,
                -1.00000000e-01,
                1.67000000e00,
                -1.00000000e-01,
                -1.00000000e-01,
                -1.00000000e-01,
                3.74540119e-03,
                5.98658484e-03,
                5.80836122e-04,
                9.50714306e-03,
                1.56018640e-03,
                8.66176146e-03,
                7.31993942e-03,
                1.55994520e-03,
                6.01115012e-03,
            ]
        )

    np.testing.assert_almost_equal(ocp.v.init.init, expected[:, np.newaxis])

    with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
        ocp.update_bounds(x_init, u_init)
