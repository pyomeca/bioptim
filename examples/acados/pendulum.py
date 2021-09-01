"""
TODO: Please confirm this example
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This simple example is a good place to start investigating bioptim using ACADOS as it describes the most common
dynamics out there (the joint torque driven), it defines an objective function and some boundaries and initial guesses
"""

import biorbd_casadi as biorbd
import numpy as np
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    Node,
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str, final_time: float, n_shooting: int, use_sx: bool = True
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    use_sx: bool
        If the ocp should be built with SX. Please note that ACADOS requires SX

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()

    target = np.zeros((nq + nqdot, 1))
    target[1, 0] = 3.14

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100.0, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1.0, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1.0, multi_thread=False)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, weight=50000, key="q", target=target[:nq, :], multi_thread=False
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, weight=50000, key="qdot", target=target[nq:, :], multi_thread=False
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0][:, 0] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (nq + nqdot))

    # Define control path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()
    torque_min, torque_max, torque_init = -100, 100, 0
    u_bounds = BoundsList()
    u_bounds.add([torque_min] * n_tau, [torque_max] * n_tau)
    u_bounds[0][n_tau - 1, :] = 0

    u_init = InitialGuessList()
    u_init.add([torque_init] * n_tau)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        use_sx=use_sx,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization using ACADOS and animates it
    """

    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=1, n_shooting=100)

    # --- Solve the program --- #
    sol = ocp.solve(solver=Solver.ACADOS)
    sol.print()

    # --- Show results --- #
    sol.print()
    sol.graphs()
    sol.animate()


if __name__ == "__main__":
    main()
