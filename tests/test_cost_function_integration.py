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
    Solver,
)
import biorbd_casadi as biorbd


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    integration_rule: IntegralApproximation,
    control_type: ControlType,
    objective: str,
    target: np.array = None,
    ode_solver: OdeSolver = OdeSolver.RK4(),
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
    ode_solver: OdeSolver = OdeSolver.RK4()
        Which type of OdeSolver to use

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
    if objective == "mayer":
        objective_functions = Objective(
            ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", integration_rule=integration_rule, target=target
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
        ode_solver=ode_solver,
        use_sx=True,
        n_threads=1,
        control_type=control_type,
    )


def sum_cost_function_output(sol):
    """
    Sum the cost function output from sol.print_cost()
    """
    captured_output = io.StringIO()  # Create StringIO object
    sys.stdout = captured_output  # and redirect stdout.
    sol.print_cost()  # Call function.
    sys.stdout = sys.__stdout__  # Reset redirect.
    idx = captured_output.getvalue().find("Sum cost functions")
    output = captured_output.getvalue()[idx:].split("\n")[0]
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
        IntegralApproximation.TRUE_TRAPEZOIDAL,
    ],
)
def test_pendulum(control_type, integration_rule, objective):
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_shooting=30,
        integration_rule=integration_rule,
        objective=objective,
        control_type=control_type,
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(5)
    sol = ocp.solve(solver)
    j_printed = sum_cost_function_output(sol)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if integration_rule == IntegralApproximation.RECTANGLE:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 36.077211633874164)
                np.testing.assert_almost_equal(j_printed, 36.077211633874164)
            else:
                np.testing.assert_almost_equal(f[0, 0], 18.91863487850207)
                np.testing.assert_almost_equal(j_printed, 18.91863487850207)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 52.0209218166193)
                np.testing.assert_almost_equal(j_printed, 52.0209218166193)
            else:
                np.testing.assert_almost_equal(f[0, 0], 18.844221574687065)
                np.testing.assert_almost_equal(j_printed, 18.844221574687065)
    elif integration_rule == IntegralApproximation.TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 36.077211633874164)
                np.testing.assert_almost_equal(j_printed, 36.077211633874164)
            else:
                np.testing.assert_almost_equal(f[0, 0], 18.91863487850206)
                np.testing.assert_almost_equal(j_printed, 18.91863487850206)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 26.170949218870444)
                np.testing.assert_almost_equal(j_printed, 26.170949218870444)
            else:
                np.testing.assert_almost_equal(f[0, 0], 18.844221574687094)
                np.testing.assert_almost_equal(j_printed, 18.844221574687094)
    elif integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 36.077211633874164)
                np.testing.assert_almost_equal(j_printed, 36.077211633874164)
            else:
                np.testing.assert_almost_equal(f[0, 0], 17.944878542423062)
                np.testing.assert_almost_equal(j_printed, 17.944878542423062)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 26.170949218870444)
                np.testing.assert_almost_equal(j_printed, 26.170949218870444)
            else:
                np.testing.assert_almost_equal(f[0, 0], 18.799673213312587)
                np.testing.assert_almost_equal(j_printed, 18.799673213312587)


@pytest.mark.parametrize(
    "objective",
    [
        "torque",
        "qdot",
    ],
)
@pytest.mark.parametrize(
    "control_type",
    [ControlType.CONSTANT],
)
@pytest.mark.parametrize(
    "integration_rule",
    [
        IntegralApproximation.RECTANGLE,
        IntegralApproximation.TRAPEZOIDAL,
        IntegralApproximation.TRUE_TRAPEZOIDAL,
    ],
)
def test_pendulum_collocation(control_type, integration_rule, objective):
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_shooting=10,
        integration_rule=integration_rule,
        objective=objective,
        control_type=control_type,
        ode_solver=OdeSolver.COLLOCATION(),
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(5)
    sol = ocp.solve(solver)
    j_printed = sum_cost_function_output(sol)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if integration_rule == IntegralApproximation.RECTANGLE:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 11.795040652982767)
                np.testing.assert_almost_equal(j_printed, 11.795040652982767)
            else:
                np.testing.assert_almost_equal(f[0, 0], 12.336208562756555)
                np.testing.assert_almost_equal(j_printed, 12.336208562756553)
    elif integration_rule == IntegralApproximation.TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 11.795040652982767)
                np.testing.assert_almost_equal(j_printed, 11.795040652982767)
            else:
                np.testing.assert_almost_equal(f[0, 0], 12.336208562756559)
                np.testing.assert_almost_equal(j_printed, 12.336208562756559)
    elif integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 11.795040652982767)
                np.testing.assert_almost_equal(j_printed, 11.795040652982767)
            else:
                np.testing.assert_almost_equal(f[0, 0], 12.336208562756564)
                np.testing.assert_almost_equal(j_printed, 12.336208562756564)


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
        IntegralApproximation.TRUE_TRAPEZOIDAL,
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
                    5.07107824,
                    5.29716282,
                    3.71599869,
                    2.69356547,
                    1.98544214,
                    1.43359131,
                    0.95250526,
                    0.48160389,
                    -0.0405487,
                    -0.70479343,
                    -1.70404096,
                    -3.67277459,
                    -12.07505107,
                    -12.48951239,
                    -7.04539771,
                    -4.69043379,
                    -3.210717,
                    -2.12953066,
                    -1.27300752,
                    -0.55915105,
                    0.0572217,
                    0.62354422,
                    1.18703801,
                    1.78233074,
                    2.45897744,
                    3.30436539,
                    4.51012445,
                    6.61008457,
                    9.64613529,
                ],
                [
                    0.0,
                    -5.09151461,
                    -5.289256,
                    -3.6730883,
                    -2.5999572,
                    -1.81858156,
                    -1.17533959,
                    -0.59215646,
                    -0.01743249,
                    0.59950305,
                    1.33570276,
                    2.36400437,
                    4.28266505,
                    12.4836082,
                    13.08487663,
                    8.28918906,
                    6.50446535,
                    5.52416911,
                    4.89409004,
                    4.45661472,
                    4.14198901,
                    3.91498066,
                    3.76221089,
                    3.68972717,
                    3.71430414,
                    3.87364337,
                    4.24824349,
                    5.02964499,
                    6.77303688,
                    9.65322827,
                ],
            ]
        )
    else:
        target = np.array(
            [
                [
                    6.01549798,
                    4.06186543,
                    2.1857845,
                    1.28263471,
                    0.74306671,
                    0.32536624,
                    -0.06946552,
                    -0.5067574,
                    -1.05437558,
                    -1.81359809,
                    -2.9872725,
                    -5.14327902,
                    -11.4338948,
                    -22.01970852,
                    -6.62382791,
                    -1.42881905,
                    1.03925748,
                    2.50114717,
                    3.41435707,
                    3.95852017,
                    4.21516739,
                    4.75104952,
                    5.61267393,
                    6.2718718,
                    6.67989085,
                    6.72242315,
                    6.11617043,
                    3.99273594,
                    -3.12543577,
                    -13.68877181,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_shooting=30,
        integration_rule=integration_rule,
        objective=objective,
        control_type=control_type,
        target=target,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(5)
    sol = ocp.solve(solver)
    j_printed = sum_cost_function_output(sol)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if integration_rule == IntegralApproximation.RECTANGLE:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 47.409664872029175)
                np.testing.assert_almost_equal(j_printed, 47.409664872029175)
            else:
                np.testing.assert_almost_equal(f[0, 0], 69.3969483839429, decimal=4)  # for windows
                np.testing.assert_almost_equal(j_printed, 69.3969483839429, decimal=4)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 47.18288247657242)
                np.testing.assert_almost_equal(j_printed, 47.18288247657242)
            else:
                np.testing.assert_almost_equal(f[0, 0], 49.52081908930845)
                np.testing.assert_almost_equal(j_printed, 49.52081908930845)
    elif integration_rule == IntegralApproximation.TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 47.20218950613529)
                np.testing.assert_almost_equal(j_printed, 47.20218950613529)
            else:
                np.testing.assert_almost_equal(f[0, 0], 79.20445223944195)
                np.testing.assert_almost_equal(j_printed, 79.20445223944195)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 48.842983152427955)
                np.testing.assert_almost_equal(j_printed, 48.842983152427955)
            else:
                np.testing.assert_almost_equal(f[0, 0], 47.038431660223246)
                np.testing.assert_almost_equal(j_printed, 47.038431660223246)
    elif integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 47.20218950613529)
                np.testing.assert_almost_equal(j_printed, 47.20218950613529)
            else:
                np.testing.assert_almost_equal(f[0, 0], 33.46130228108698)
                np.testing.assert_almost_equal(j_printed, 33.46130228108698)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 48.842983152427955)
                np.testing.assert_almost_equal(j_printed, 48.842983152427955)
            else:
                np.testing.assert_almost_equal(f[0, 0], 55.5377703306112)
                np.testing.assert_almost_equal(j_printed, 55.5377703306112)


@pytest.mark.parametrize(
    "integration_rule",
    [
        IntegralApproximation.RECTANGLE,
        IntegralApproximation.TRAPEZOIDAL,
        IntegralApproximation.TRUE_TRAPEZOIDAL,
    ],
)
def test_error_mayer_trapz(integration_rule):
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    with pytest.raises(
        ValueError,
        match="Mayer objective functions cannot be integrated, "
        "remove the argument integration_rule or use a Lagrange objective function",
    ):

        ocp = prepare_ocp(
            biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
            n_shooting=30,
            integration_rule=integration_rule,
            objective="mayer",
            control_type=ControlType.LINEAR_CONTINUOUS,
        )
