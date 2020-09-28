from pathlib import Path

import pytest
import numpy as np
import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    BoundsOption,
    InitialConditionsOption,
    ParameterList,
    Bounds,
    InterpolationType,
    InitialConditionsOption,
    InitialConditions,
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

    X_bounds = BoundsOption([-np.ones((nq * 2, 1)), np.ones((nq * 2, 1))])
    U_bounds = BoundsOption([-2.0 * np.ones((nq, 1)), 2.0 * np.ones((nq, 1))])
    ocp.update_bounds(X_bounds, U_bounds)

    expected = np.append(np.tile(np.append(-np.ones((nq * 2, 1)), -2.0 * np.ones((nq, 1))), ns), -np.ones((nq * 2, 1)))
    np.testing.assert_almost_equal(ocp.V_bounds.min, expected.reshape(128, 1))
    expected = np.append(np.tile(np.append(np.ones((nq * 2, 1)), 2.0 * np.ones((nq, 1))), ns), np.ones((nq * 2, 1)))
    np.testing.assert_almost_equal(ocp.V_bounds.max, expected.reshape(128, 1))

    X_init = InitialConditionsOption(0.5 * np.ones((nq * 2, 1)))
    U_init = InitialConditionsOption(-0.5 * np.ones((nq, 1)))
    ocp.update_initial_guess(X_init, U_init)
    expected = np.append(
        np.tile(np.append(0.5 * np.ones((nq * 2, 1)), -0.5 * np.ones((nq, 1))), ns), 0.5 * np.ones((nq * 2, 1))
    )
    np.testing.assert_almost_equal(ocp.V_init.init, expected.reshape(128, 1))

    X_bounds = BoundsOption([-2.0 * np.ones((nq * 2, 1)), 2.0 * np.ones((nq * 2, 1))])
    U_bounds = BoundsOption([-4.0 * np.ones((nq, 1)), 4.0 * np.ones((nq, 1))])
    ocp.update_bounds(X_bounds=X_bounds)
    ocp.update_bounds(U_bounds=U_bounds)

    expected = np.append(
        np.tile(np.append(-2.0 * np.ones((nq * 2, 1)), -4.0 * np.ones((nq, 1))), ns), -2.0 * np.ones((nq * 2, 1))
    )
    np.testing.assert_almost_equal(ocp.V_bounds.min, expected.reshape(128, 1))
    expected = np.append(
        np.tile(np.append(2.0 * np.ones((nq * 2, 1)), 4.0 * np.ones((nq, 1))), ns), 2.0 * np.ones((nq * 2, 1))
    )
    np.testing.assert_almost_equal(ocp.V_bounds.max, expected.reshape(128, 1))

    X_init = InitialConditionsOption(0.25 * np.ones((nq * 2, 1)))
    U_init = InitialConditionsOption(-0.25 * np.ones((nq, 1)))
    ocp.update_initial_guess(X_init, U_init)
    expected = np.append(
        np.tile(np.append(0.25 * np.ones((nq * 2, 1)), -0.25 * np.ones((nq, 1))), ns), 0.25 * np.ones((nq * 2, 1))
    )
    np.testing.assert_almost_equal(ocp.V_init.init, expected.reshape(128, 1))

    with pytest.raises(
        RuntimeError, match="X_init should be built from a InitialConditionsOption or InitialConditionsList"
    ):
        ocp.update_initial_guess(X_bounds, U_bounds)
    with pytest.raises(RuntimeError, match="X_bounds should be built from a BoundsOption or BoundsList"):
        ocp.update_bounds(X_init, U_init)

def my_parameter_function(biorbd_model, value, extra_value):
    biorbd_model.setGravity(biorbd.Vector3d(0, 0, value + extra_value))

def my_target_function(ocp, value, target_value):
    return value + target_value

def test_update_bounds_and_init_with_param():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    biorbd_model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod")
    nq = biorbd_model.nbQ()
    ns = 10

    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    parameters = ParameterList()
    bounds_gravity = Bounds(min_bounds=-10, max_bounds=-6, interpolation=InterpolationType.CONSTANT)
    initial_gravity = InitialConditions(-8)
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
            extra_value=1
            )

    ocp = OptimalControlProgram(biorbd_model, dynamics, ns, 1.0, parameters=parameters)

    X_bounds = BoundsOption([-np.ones((nq * 2, 1)), np.ones((nq * 2, 1))])
    U_bounds = BoundsOption([-2.0 * np.ones((nq, 1)), 2.0 * np.ones((nq, 1))])
    ocp.update_bounds(X_bounds, U_bounds)

    expected = np.append(np.tile(np.append(-np.ones((nq * 2, 1)), -2.0 * np.ones((nq, 1))), ns), -np.ones((nq * 2, 1)))
    np.testing.assert_almost_equal(ocp.V_bounds.min, expected.reshape(128, 1))
    expected = np.append(np.tile(np.append(np.ones((nq * 2, 1)), 2.0 * np.ones((nq, 1))), ns), np.ones((nq * 2, 1)))
    np.testing.assert_almost_equal(ocp.V_bounds.max, expected.reshape(128, 1))

    X_init = InitialConditionsOption(0.5 * np.ones((nq * 2, 1)))
    U_init = InitialConditionsOption(-0.5 * np.ones((nq, 1)))
    ocp.update_initial_guess(X_init, U_init)
    expected = np.append(
        np.tile(np.append(0.5 * np.ones((nq * 2, 1)), -0.5 * np.ones((nq, 1))), ns), 0.5 * np.ones((nq * 2, 1))
    )
    np.testing.assert_almost_equal(ocp.V_init.init, expected.reshape(128, 1))
