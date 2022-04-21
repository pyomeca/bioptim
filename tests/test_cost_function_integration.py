"""
Test for file IO
"""
import os
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

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)
    for ii in range(10):
        print(integration_rule)
    # Add objective functions
    if objective == "torque":
        objective_functions = Objective(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", integration_rule=integration_rule
        )
    if objective == "qdot":
        objective_functions = Objective(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", integration_rule=integration_rule
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


@pytest.mark.parametrize(
    "objective",
    [
        "torque",
        # "qdot",
    ],
)
@pytest.mark.parametrize(
    "control_type",
    [
        ControlType.CONSTANT,
        # ControlType.LINEAR_CONTINUOUS
    ],
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
        n_shooting=20,
        integration_rule=integration_rule,
        objective=objective,
        control_type=control_type,
    )

    sol = ocp.solve()
    sol.print_cost()
    print(sol.cost)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if integration_rule == IntegralApproximation.RECTANGLE:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 695.8827067300126)
            else:
                np.testing.assert_almost_equal(f[0, 0], 33.240173228232464)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 215.60226886088248)
            else:
                np.testing.assert_almost_equal(f[0, 0], 17.123731735231377)
    elif integration_rule == IntegralApproximation.TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 695.8827067300126)
            else:
                np.testing.assert_almost_equal(f[0, 0], 33.240173228232464)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                np.testing.assert_almost_equal(f[0, 0], 215.60226886088248)
            else:
                np.testing.assert_almost_equal(f[0, 0], 17.123731735231377)
