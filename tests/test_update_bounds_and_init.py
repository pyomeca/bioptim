from pathlib import Path

import pytest
import numpy as np
import biorbd

from bioptim import (
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    BoundsOption,
    InitialGuessOption,
    ParameterList,
    Bounds,
    InterpolationType,
    InitialGuessOption,
    InitialGuess,
    ObjectiveOption,
    Objective,
)


def test_double_update_bounds_and_init():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    biorbd_model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod")
    nq = biorbd_model.nbQ()
    ns = 10

    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, ns, 1.0)

    x_bounds = BoundsOption([-np.ones((nq * 2, 1)), np.ones((nq * 2, 1))])
    u_bounds = BoundsOption([-2.0 * np.ones((nq, 1)), 2.0 * np.ones((nq, 1))])
    ocp.update_bounds(x_bounds, u_bounds)

    expected = np.append(np.tile(np.append(-np.ones((nq * 2, 1)), -2.0 * np.ones((nq, 1))), ns), -np.ones((nq * 2, 1)))
    np.testing.assert_almost_equal(ocp.V_bounds.min, expected.reshape(128, 1))
    expected = np.append(np.tile(np.append(np.ones((nq * 2, 1)), 2.0 * np.ones((nq, 1))), ns), np.ones((nq * 2, 1)))
    np.testing.assert_almost_equal(ocp.V_bounds.max, expected.reshape(128, 1))

    x_init = InitialGuessOption(0.5 * np.ones((nq * 2, 1)))
    u_init = InitialGuessOption(-0.5 * np.ones((nq, 1)))
    ocp.update_initial_guess(x_init, u_init)
    expected = np.append(
        np.tile(np.append(0.5 * np.ones((nq * 2, 1)), -0.5 * np.ones((nq, 1))), ns), 0.5 * np.ones((nq * 2, 1))
    )
    np.testing.assert_almost_equal(ocp.V_init.init, expected.reshape(128, 1))

    x_bounds = BoundsOption([-2.0 * np.ones((nq * 2, 1)), 2.0 * np.ones((nq * 2, 1))])
    u_bounds = BoundsOption([-4.0 * np.ones((nq, 1)), 4.0 * np.ones((nq, 1))])
    ocp.update_bounds(x_bounds=x_bounds)
    ocp.update_bounds(u_bounds=u_bounds)

    expected = np.append(
        np.tile(np.append(-2.0 * np.ones((nq * 2, 1)), -4.0 * np.ones((nq, 1))), ns), -2.0 * np.ones((nq * 2, 1))
    )
    np.testing.assert_almost_equal(ocp.V_bounds.min, expected.reshape(128, 1))
    expected = np.append(
        np.tile(np.append(2.0 * np.ones((nq * 2, 1)), 4.0 * np.ones((nq, 1))), ns), 2.0 * np.ones((nq * 2, 1))
    )
    np.testing.assert_almost_equal(ocp.V_bounds.max, expected.reshape(128, 1))

    x_init = InitialGuessOption(0.25 * np.ones((nq * 2, 1)))
    u_init = InitialGuessOption(-0.25 * np.ones((nq, 1)))
    ocp.update_initial_guess(x_init, u_init)
    expected = np.append(
        np.tile(np.append(0.25 * np.ones((nq * 2, 1)), -0.25 * np.ones((nq, 1))), ns), 0.25 * np.ones((nq * 2, 1))
    )
    np.testing.assert_almost_equal(ocp.V_init.init, expected.reshape(128, 1))

    with pytest.raises(RuntimeError, match="x_init should be built from a InitialGuessOption or InitialGuessList"):
        ocp.update_initial_guess(x_bounds, u_bounds)
    with pytest.raises(RuntimeError, match="x_bounds should be built from a BoundsOption or BoundsList"):
        ocp.update_bounds(x_init, u_init)


def test_update_bounds_and_init_with_param():
    def my_parameter_function(biorbd_model, value, extra_value):
        biorbd_model.setGravity(biorbd.Vector3d(0, 0, value + extra_value))

    def my_target_function(ocp, value, target_value):
        return value + target_value

    PROJECT_FOLDER = Path(__file__).parent / ".."
    biorbd_model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod")
    nq = biorbd_model.nbQ()
    ns = 10
    g_min, g_max, g_init = -10, -6, -8

    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    parameters = ParameterList()
    bounds_gravity = Bounds(min_bound=g_min, max_bound=g_max, interpolation=InterpolationType.CONSTANT)
    initial_gravity = InitialGuess(g_init)
    parameter_objective_functions = ObjectiveOption(
        my_target_function, weight=10, quadratic=True, custom_type=Objective.Parameter, target_value=-8
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

    x_bounds = BoundsOption([-np.ones((nq * 2, 1)), np.ones((nq * 2, 1))])
    u_bounds = BoundsOption([-2.0 * np.ones((nq, 1)), 2.0 * np.ones((nq, 1))])
    ocp.update_bounds(x_bounds, u_bounds)

    expected = np.append(np.tile(np.append(-np.ones((nq * 2, 1)), -2.0 * np.ones((nq, 1))), ns), -np.ones((nq * 2, 1)))
    np.testing.assert_almost_equal(ocp.V_bounds.min, np.append([g_min], expected).reshape(129, 1))
    expected = np.append(np.tile(np.append(np.ones((nq * 2, 1)), 2.0 * np.ones((nq, 1))), ns), np.ones((nq * 2, 1)))
    np.testing.assert_almost_equal(ocp.V_bounds.max, np.append([[g_max]], expected).reshape(129, 1))

    x_init = InitialGuessOption(0.5 * np.ones((nq * 2, 1)))
    u_init = InitialGuessOption(-0.5 * np.ones((nq, 1)))
    ocp.update_initial_guess(x_init, u_init)
    expected = np.append(
        np.tile(np.append(0.5 * np.ones((nq * 2, 1)), -0.5 * np.ones((nq, 1))), ns), 0.5 * np.ones((nq * 2, 1))
    )
    np.testing.assert_almost_equal(ocp.V_init.init, np.append([g_init], expected).reshape(129, 1))


def test_add_wrong_param():
    g_min, g_max, g_init = -10, -6, -8

    def my_parameter_function(biorbd_model, value, extra_value):
        biorbd_model.setGravity(biorbd.Vector3d(0, 0, value + extra_value))

    def my_target_function(ocp, value, target_value):
        return value + target_value

    parameters = ParameterList()
    initial_gravity = InitialGuess(g_init)
    bounds_gravity = Bounds(min_bound=g_min, max_bound=g_max, interpolation=InterpolationType.CONSTANT)
    parameter_objective_functions = ObjectiveOption(
        my_target_function, weight=10, quadratic=True, custom_type=Objective.Parameter, target_value=-8
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
            InitialGuess(),
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
            Bounds(),
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
