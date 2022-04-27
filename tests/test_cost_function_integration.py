"""
Test for file IO
"""
import os
import sys
import io
import pytest
import numpy as np
from bioptim import (
    OdeSolver,
    ControlType,
    IntegralApproximation,
    OptimalControlProgram,
    QAndQDotBounds,
    Objective,
    ObjectiveFcn,
    Dynamics,
    DynamicsFcn,
    InitialGuess,
    Bounds,
)
import biorbd_casadi as biorbd


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    integration_rule: IntegralApproximation,
    control_type: ControlType,
    objective: str,
    target: np.array = None,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    integration_rule: IntegralApproximation
        The integration rule to use
    control_type: ControlType
        The type of control to use (constant or linear)
    objective: str
        The objective to minimize (torque or power)
    target: np.array
        The target value to reach

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    if objective == "torque":
        objective_functions = Objective(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", integration_rule=integration_rule, target=target
        )
    if objective == "qdot":
        objective_functions = Objective(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", integration_rule=integration_rule, target=target
        )

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Initial guess
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0  # Prevent the model from actively rotate

    u_init = InitialGuess([tau_init] * n_tau)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        1,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=OdeSolver.RK4(),
        use_sx=True,
        n_threads=1,
        control_type=control_type,
    )


def sum_cost_function_output(sol):
    """
    Sum the cost function output from sol.print_cost()
    """
    capturedOutput = io.StringIO()  # Create StringIO object
    sys.stdout = capturedOutput  # and redirect stdout.
    sol.print_cost()  # Call function.
    sys.stdout = sys.__stdout__  # Reset redirect.
    output = capturedOutput.getvalue()
    idx = capturedOutput.getvalue().find("Sum cost functions")
    output = capturedOutput.getvalue()[idx:].split("\n")[0]
    idx = len("Sum cost functions: ")
    return float(output[idx:])


@pytest.mark.parametrize(
    "objective",
    [
        "torque",
        "qdot",
    ],
)
@pytest.mark.parametrize(
    "control_type",
    [ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS],
)
@pytest.mark.parametrize(
    "integration_rule",
    [
        IntegralApproximation.RECTANGLE,
        IntegralApproximation.TRAPEZOIDAL,
    ],
)
def test_pendulum(control_type, integration_rule, objective):
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_shooting=15,
        integration_rule=integration_rule,
        objective=objective,
        control_type=control_type,
    )

    sol = ocp.solve()
    j_printed = sum_cost_function_output(sol)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if integration_rule == IntegralApproximation.RECTANGLE:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 84.49740183326932)
                np.testing.assert_almost_equal(j_printed, 84.49740183326932)
            else:
                np.testing.assert_almost_equal(f[0, 0], 30.74653667406686)
                np.testing.assert_almost_equal(j_printed, 30.74653667406686)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 55.84321417560313)
                np.testing.assert_almost_equal(j_printed, 55.84321417560313)
            else:
                np.testing.assert_almost_equal(f[0, 0], 23.371290063644963)
                np.testing.assert_almost_equal(j_printed, 23.371290063644963)
    elif integration_rule == IntegralApproximation.TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 84.49740183326932)
                np.testing.assert_almost_equal(j_printed, 84.49740183326932)
            else:
                np.testing.assert_almost_equal(f[0, 0], 32.386868834009924)
                np.testing.assert_almost_equal(j_printed, 32.386868834009924)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 50.47034448982324)
                np.testing.assert_almost_equal(j_printed, 50.47034448982324)
            else:
                np.testing.assert_almost_equal(f[0, 0], 23.371290063644935)
                np.testing.assert_almost_equal(j_printed, 23.371290063644935)


@pytest.mark.parametrize(
    "objective",
    [
        "torque",
        "qdot",
    ],
)
@pytest.mark.parametrize(
    "control_type",
    [ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS],
)
@pytest.mark.parametrize(
    "integration_rule",
    [
        IntegralApproximation.RECTANGLE,
        IntegralApproximation.TRAPEZOIDAL,
    ],
)
def test_pendulum_target(control_type, integration_rule, objective):
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    if objective == "qdot":
        target = np.array(
            [
                [
                    0.0,
                    6.51208486,
                    3.27752055,
                    1.86135695,
                    0.96701456,
                    0.18193208,
                    -0.83287956,
                    -2.9622631,
                    -23.7302095,
                    -6.06721842,
                    -2.65133032,
                    -0.72913491,
                    0.68722248,
                    2.74211658,
                    8.90941496,
                ],
                [
                    0.0,
                    -6.49835123,
                    -3.24407569,
                    -1.71133185,
                    -0.6099655,
                    0.407523,
                    1.59240089,
                    3.68457442,
                    23.79597936,
                    7.50801022,
                    5.38743813,
                    4.5751776,
                    4.26241442,
                    4.73597801,
                    9.15226233,
                ],
            ]
        )
    else:
        target = np.array(
            [
                [
                    7.69288937,
                    2.15792818,
                    -0.30564323,
                    -1.12922158,
                    -1.64495223,
                    -2.42867254,
                    -4.39543379,
                    -17.60401641,
                    -6.637938,
                    3.8367779,
                    4.64704813,
                    4.29142416,
                    11.04939719,
                    18.56698167,
                    -17.28621612,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_shooting=15,
        integration_rule=integration_rule,
        objective=objective,
        control_type=control_type,
        target=target,
    )

    sol = ocp.solve()
    j_printed = sum_cost_function_output(sol)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if integration_rule == IntegralApproximation.RECTANGLE:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 36.839829039151184)
                np.testing.assert_almost_equal(j_printed, 36.839829039151184)
            else:
                np.testing.assert_almost_equal(f[0, 0], 6.313081884718639e-07)
                np.testing.assert_almost_equal(j_printed, 6.313081884718639e-07)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 264.0066555609535)
                np.testing.assert_almost_equal(j_printed, 264.0066555609535)
            else:
                np.testing.assert_almost_equal(f[0, 0], 71.44276846089537)
                np.testing.assert_almost_equal(j_printed, 71.44276846089537)
    elif integration_rule == IntegralApproximation.TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 379.0813647567205)
                np.testing.assert_almost_equal(j_printed, 379.0813647567205)
            else:
                np.testing.assert_almost_equal(f[0, 0], 43.015543997237025)
                np.testing.assert_almost_equal(j_printed, 43.015543997237025)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 538.8013374888595)
                np.testing.assert_almost_equal(j_printed, 538.8013374888595)
            else:
                np.testing.assert_almost_equal(f[0, 0], 66.07715305531984)
                np.testing.assert_almost_equal(j_printed, 66.07715305531984)
