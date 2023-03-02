import pytest
import numpy as np
import biorbd_casadi as biorbd
from casadi import MX
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsFcn,
    DynamicsList,
    Bounds,
    ParameterList,
    InterpolationType,
    InitialGuess,
    NoisedInitialGuess,
    Objective,
    ObjectiveFcn,
    OdeSolver,
    MagnitudeType,
)

from .utils import TestUtils


def test_double_update_bounds_and_init():
    bioptim_folder = TestUtils.bioptim_folder()
    bio_model = BiorbdModel(bioptim_folder + "/examples/track/models/cube_and_line.bioMod")
    nq = bio_model.nb_q
    ns = 10

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    x_init = InitialGuess([0] * (bio_model.nb_q + bio_model.nb_qdot))
    u_init = InitialGuess([0] * bio_model.nb_tau)
    ocp = OptimalControlProgram(bio_model, dynamics, ns, 1.0, x_init=x_init, u_init=u_init)

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
    def my_parameter_function(bio_model, value, extra_value):
        new_gravity = MX.zeros(3, 1)
        new_gravity[2] = value + extra_value
        bio_model.set_gravity(new_gravity)

    def my_target_function(ocp, value, target_value):
        return value + target_value

    bio_model = BiorbdModel(TestUtils.bioptim_folder() + "/examples/track/models/cube_and_line.bioMod")
    nq = bio_model.nb_q
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

    x_init = InitialGuess([0] * (bio_model.nb_q + bio_model.nb_tau))
    u_init = InitialGuess([0] * bio_model.nb_tau)
    ocp = OptimalControlProgram(bio_model, dynamics, ns, 1.0, parameters=parameters, x_init=x_init, u_init=u_init)

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

    def my_parameter_function(bio_model, value, extra_value):
        bio_model.set_gravity(biorbd.Vector3d(0, 0, value + extra_value))

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
    bio_model = BiorbdModel(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = bio_model.nb_q
    nqdot = bio_model.nb_qdot
    ntau = bio_model.nb_tau
    ns = 3
    phase_time = 1.0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    x_init = InitialGuess([0] * (bio_model.nb_q + bio_model.nb_qdot))
    u_init = InitialGuess([0] * bio_model.nb_tau)
    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting=ns,
        phase_time=phase_time,
        ode_solver=OdeSolver.RK4(),
        x_init=x_init,
        u_init=u_init,
    )

    # Path constraint and control path constraints
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
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
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns + 1,
        bound_push=0.1,
        **extra_params_x,
    )
    u_init = NoisedInitialGuess(
        u,
        t=t,
        interpolation=interpolation,
        bounds=u_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns,
        bound_push=0.1,
        **extra_params_u,
    )

    if interpolation == InterpolationType.ALL_POINTS:
        with pytest.raises(ValueError, match="InterpolationType.ALL_POINTS must only be used with direct collocation"):
            ocp.update_initial_guess(x_init, u_init)
    else:
        ocp.update_initial_guess(x_init, u_init)

        if interpolation == InterpolationType.CONSTANT:
            expected = np.array(
                [
                    [0.00292881],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.01291136],
                    [0.00583576],
                    [-0.01464717],
                    [0.53482051],
                    [0.41798243],
                    [0.37593374],
                    [0.0061658],
                    [-0.00249651],
                    [0.03665925],
                    [-0.53905199],
                    [0.34954208],
                    [-0.04840646],
                    [0.00269299],
                    [0.0],
                    [1.67],
                    [0.0],
                    [0.0],
                    [0.0],
                    [-1.5269023],
                    [1.77867567],
                    [-0.94177755],
                    [0.55968409],
                    [0.08739329],
                    [1.09693476],
                    [-1.42658685],
                    [-0.34135224],
                    [-0.17539867],
                ]
            )

        elif interpolation == InterpolationType.LINEAR:
            expected = np.array(
                [
                    [1.00292881e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [1.26291136e00],
                    [5.83576452e-03],
                    [3.77852829e-01],
                    [5.34820509e-01],
                    [4.17982425e-01],
                    [3.75933739e-01],
                    [1.50616580e00],
                    [-2.49651155e-03],
                    [8.21659249e-01],
                    [-5.39051987e-01],
                    [3.49542082e-01],
                    [-4.84064610e-02],
                    [1.75269299e00],
                    [0.00000000e00],
                    [1.67000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [-7.69022965e-02],
                    [1.15886757e01],
                    [1.33822245e00],
                    [1.04301742e00],
                    [9.89739329e00],
                    [1.85693476e00],
                    [-1.90992018e00],
                    [9.46864776e00],
                    [-9.35398671e-01],
                ]
            )

        elif interpolation == InterpolationType.SPLINE:
            expected = np.array(
                [
                    [0.61502453],
                    [-0.1],
                    [-0.1],
                    [-0.1],
                    [-0.1],
                    [-0.1],
                    [0.76732112],
                    [0.43533947],
                    [0.19273203],
                    [1.15035619],
                    [0.76088147],
                    [0.81430269],
                    [0.90922359],
                    [0.13708974],
                    [0.32886699],
                    [-0.32665397],
                    [0.78933071],
                    [0.15428881],
                    [0.57293724],
                    [-0.1],
                    [1.67],
                    [-0.1],
                    [-0.1],
                    [-0.1],
                    [-0.70590907],
                    [2.24732687],
                    [-0.65897059],
                    [1.08074616],
                    [0.85131915],
                    [1.31781816],
                    [-1.21759843],
                    [0.30813958],
                    [-0.03112013],
                ]
            )

        elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            expected = np.array(
                [
                    [1.00292881e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [1.51291136e00],
                    [5.83576452e-03],
                    [7.70352829e-01],
                    [5.34820509e-01],
                    [4.17982425e-01],
                    [3.75933739e-01],
                    [1.50616580e00],
                    [-2.49651155e-03],
                    [8.21659249e-01],
                    [-5.39051987e-01],
                    [3.49542082e-01],
                    [-4.84064610e-02],
                    [1.50269299e00],
                    [0.00000000e00],
                    [1.67000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [-7.69022965e-02],
                    [1.15886757e01],
                    [1.33822245e00],
                    [5.59684085e-01],
                    [9.89739329e00],
                    [1.09693476e00],
                    [-1.42658685e00],
                    [9.46864776e00],
                    [-1.75398671e-01],
                ]
            )

        if interpolation == InterpolationType.EACH_FRAME:
            expected = np.array(
                [
                    [0.00292881],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.01291136],
                    [0.00583576],
                    [-0.01464717],
                    [0.53482051],
                    [0.41798243],
                    [0.37593374],
                    [0.0061658],
                    [-0.00249651],
                    [0.03665925],
                    [-0.53905199],
                    [0.34954208],
                    [-0.04840646],
                    [0.00269299],
                    [0.0],
                    [1.67],
                    [0.0],
                    [0.0],
                    [0.0],
                    [-1.5269023],
                    [1.77867567],
                    [-0.94177755],
                    [0.55968409],
                    [0.08739329],
                    [1.09693476],
                    [-1.42658685],
                    [-0.34135224],
                    [-0.17539867],
                ]
            )

        np.testing.assert_almost_equal(ocp.v.init.init, expected)

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
    bio_model = BiorbdModel(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = bio_model.nb_q
    nqdot = bio_model.nb_qdot
    ntau = bio_model.nb_tau
    ns = 3
    phase_time = 1.0
    solver = OdeSolver.COLLOCATION(polynomial_degree=1)
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    x_init = InitialGuess([0] * (bio_model.nb_q + bio_model.nb_qdot))
    u_init = InitialGuess([0] * bio_model.nb_tau)
    ocp = OptimalControlProgram(
        bio_model, dynamics, n_shooting=ns, phase_time=phase_time, ode_solver=solver, x_init=x_init, u_init=u_init
    )

    # Path constraint and control path constraints
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
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
        for i in range(nq + nqdot):
            x[i, :] = np.linspace(i, i + 1, ns + 1)
        u = np.zeros((ntau, ns))
    elif interpolation == InterpolationType.ALL_POINTS:
        x = np.zeros((nq + nqdot, ns * (solver.polynomial_degree + 1) + 1))
        for i in range(nq + nqdot):
            x[i, :] = np.linspace(i, i + 1, ns * (solver.polynomial_degree + 1) + 1)
        u = np.zeros((ntau, ns))
    elif interpolation == InterpolationType.SPLINE:
        # Bound , assume the first and last point are 0 and final respectively
        t = np.hstack((0, np.sort(np.random.random((3,)) * phase_time), phase_time))
        x = np.random.random((nq + nqdot, 5))
        u = np.random.random((ntau, 5))
    else:
        raise NotImplementedError("This interpolation is not implemented yet")

    np.random.seed(0)
    x_init = NoisedInitialGuess(
        initial_guess=x,
        t=t,
        interpolation=interpolation,
        bounds=x_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns + 1,
        bound_push=0.1,
        **extra_params_x,
    )
    u_init = NoisedInitialGuess(
        u,
        t=t,
        interpolation=interpolation,
        bounds=u_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns,
        bound_push=0.1,
        **extra_params_u,
    )

    with pytest.raises(
        NotImplementedError,
        match="It is not possible to use initial guess with NoisedInitialGuess as it won't produce the expected randomness",
    ):
        ocp.update_initial_guess(x_init, u_init)

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
    bio_model = BiorbdModel(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = bio_model.nb_q
    nqdot = bio_model.nb_qdot
    ntau = bio_model.nb_tau
    ns = 3
    phase_time = 1.0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(bio_model, dynamics, n_shooting=ns, phase_time=phase_time)

    # Path constraint and control path constraints
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
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
        x = np.zeros((nq * 2, ns + 1))
        for i in range(ns + 1):
            x[i, :] = np.linspace(i, i + 1, ns + 1)
        x = InitialGuess(x, interpolation=interpolation)
        u = InitialGuess(np.zeros((ntau, ns)), interpolation=interpolation)
    elif interpolation == InterpolationType.ALL_POINTS:
        x = np.zeros((nq * 2, ns + 1))
        for i in range(ns + 1):
            x[i, :] = np.linspace(i, i + 1, ns + 1)
        x = InitialGuess(x, interpolation=interpolation)
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
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns + 1,
        bound_push=0.1,
        seed=42,
        **extra_params_x,
    )
    u_init = NoisedInitialGuess(
        initial_guess=u,
        bounds=u_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns,
        bound_push=0.1,
        seed=42,
        **extra_params_u,
    )
    if interpolation == InterpolationType.ALL_POINTS:
        with pytest.raises(ValueError, match="InterpolationType.ALL_POINTS must only be used with direct collocation"):
            ocp.update_initial_guess(x_init, u_init)
    else:
        ocp.update_initial_guess(x_init, u_init)

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
                    1.14685476,
                    0.9,
                    2.34640692,
                    3.15259067,
                    0.01555492,
                    -0.22651269,
                    1.47362648,
                    0.9,
                    2.6365441,
                    3.4667514,
                    -0.04276021,
                    -0.13059937,
                    1.80295975,
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


@pytest.mark.parametrize("n_extra", [0, 1])
def test_update_noised_initial_guess_rk4(n_extra):
    bioptim_folder = TestUtils.bioptim_folder()
    bio_model = BiorbdModel(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = bio_model.nb_q
    nqdot = bio_model.nb_qdot
    ntau = bio_model.nb_tau
    ns = 3
    phase_time = 1.0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    x_init = InitialGuess([0] * (bio_model.nb_q + bio_model.nb_qdot))
    u_init = InitialGuess([0] * bio_model.nb_tau)
    ocp = OptimalControlProgram(bio_model, dynamics, n_shooting=ns, phase_time=phase_time, x_init=x_init, u_init=u_init)

    # Path constraint and control path constraints
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    tau_min, tau_max, tau_init = -100, 100, 0.3
    u_bounds = Bounds([tau_min] * ntau, [tau_max] * ntau)

    # Initial guesses
    t = None
    extra_params_x = {}
    extra_params_u = {}
    x = InitialGuess([1] * (nq + nqdot), interpolation=InterpolationType.CONSTANT)
    u = InitialGuess([tau_init] * ntau, interpolation=InterpolationType.CONSTANT)

    state_noise = np.array([0.01] * nq + [0.2] * nqdot + [0.1] * n_extra)
    if n_extra > 0:
        with pytest.raises(
            ValueError, match="magnitude must be a float or list of float of the size of states or controls"
        ):
            NoisedInitialGuess(
                initial_guess=x,
                bounds=x_bounds,
                magnitude=state_noise,
                magnitude_type=MagnitudeType.RELATIVE,
                n_shooting=ns + 1,
                bound_push=0.1,
                seed=42,
                **extra_params_x,
            )
        return
    else:
        x_init = NoisedInitialGuess(
            initial_guess=x,
            bounds=x_bounds,
            magnitude=state_noise,
            magnitude_type=MagnitudeType.RELATIVE,
            n_shooting=ns + 1,
            bound_push=0.1,
            seed=42,
            **extra_params_x,
        )

    u_init = NoisedInitialGuess(
        initial_guess=u,
        bounds=u_bounds,
        magnitude=np.array([0.03] * ntau),
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns,
        bound_push=0.1,
        seed=42,
        **extra_params_u,
    )

    ocp.update_initial_guess(x_init, u_init)
    print(ocp.v.init.init)
    expected = np.array(
        [
            [0.99247241],
            [-0.1],
            [-0.1],
            [-0.1],
            [-0.1],
            [-0.1],
            [1.02704286],
            [0.98623978],
            [1.02614717],
            [-6.22970669],
            [1.62219699],
            [-8.06050751],
            [1.01391964],
            [0.98232334],
            [0.93975487],
            [-6.99661076],
            [-0.71040824],
            [-4.22397476],
            [1.00591951],
            [-0.1],
            [1.67],
            [-0.1],
            [-0.1],
            [-0.1],
            [-1.20551857],
            [1.48390181],
            [-5.00299665],
            [5.70857168],
            [-3.82777631],
            [4.69411375],
            [3.0839273],
            [-3.82806576],
            [1.51338014],
        ]
    )

    np.testing.assert_almost_equal(ocp.v.init.init, expected)

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
    bio_model = BiorbdModel(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = bio_model.nb_q
    nqdot = bio_model.nb_qdot
    ntau = bio_model.nb_tau
    ns = 3
    phase_time = 1.0
    solver = OdeSolver.COLLOCATION(polynomial_degree=1)

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    x_init = InitialGuess([0] * (bio_model.nb_q + bio_model.nb_qdot))
    u_init = InitialGuess([0] * bio_model.nb_tau)
    ocp = OptimalControlProgram(
        bio_model, dynamics, n_shooting=ns, phase_time=phase_time, ode_solver=solver, x_init=x_init, u_init=u_init
    )

    # Path constraint and control path constraints
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
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
        x = np.zeros((nq * 2, ns + 1))
        for i in range(nq * 2):
            x[i, :] = np.linspace(0, 1, ns + 1)
        x = InitialGuess(x, interpolation=interpolation)
        u = InitialGuess(np.zeros((ntau, ns)), interpolation=interpolation)
    elif interpolation == InterpolationType.ALL_POINTS:
        x = np.zeros((nq * 2, ns * (solver.polynomial_degree + 1) + 1))
        for i in range(nq * 2):
            x[i, :] = np.linspace(0, 1, ns * (solver.polynomial_degree + 1) + 1)
        x = InitialGuess(x, interpolation=interpolation)
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
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns + 1,
        bound_push=0.1,
        seed=42,
        **extra_params_x,
    )
    u_init = NoisedInitialGuess(
        initial_guess=u,
        bounds=u_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns,
        bound_push=0.1,
        seed=42,
        **extra_params_u,
    )

    with pytest.raises(
        NotImplementedError,
        match="It is not possible to use initial guess with NoisedInitialGuess as it won't produce the expected randomness",
    ):
        ocp.update_initial_guess(x_init, u_init)

    with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
        ocp.update_bounds(x_init, u_init)


@pytest.mark.parametrize(
    "interpolation",
    [
        InterpolationType.CONSTANT,
    ],
)
def test_update_noised_initial_guess_list(interpolation):
    bioptim_folder = TestUtils.bioptim_folder()
    bio_model = BiorbdModel(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = bio_model.nb_q
    nqdot = bio_model.nb_qdot
    ntau = bio_model.nb_tau
    ns = 3
    phase_time = 1.0
    solver = OdeSolver.COLLOCATION(polynomial_degree=1)

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    x_init = InitialGuess([0] * (bio_model.nb_q + bio_model.nb_qdot))
    u_init = InitialGuess([0] * bio_model.nb_tau)
    ocp = OptimalControlProgram(
        bio_model, dynamics, n_shooting=ns, phase_time=phase_time, ode_solver=solver, x_init=x_init, u_init=u_init
    )

    # Path constraint and control path constraints
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * ntau, [tau_max] * ntau)

    x_init = NoisedInitialGuess(
        initial_guess=[0] * (nq + nqdot),
        interpolation=InterpolationType.CONSTANT,
        bounds=x_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns + 1,
        bound_push=0.1,
        seed=42,
    )
    u_init = NoisedInitialGuess(
        initial_guess=[tau_init] * ntau,
        interpolation=InterpolationType.CONSTANT,
        bounds=u_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns,
        bound_push=0.1,
        seed=42,
    )

    with pytest.raises(
        NotImplementedError,
        match="It is not possible to use initial guess with NoisedInitialGuess as it won't produce the expected randomness",
    ):
        ocp.update_initial_guess(x_init, u_init)

    with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
        ocp.update_bounds(x_init, u_init)
