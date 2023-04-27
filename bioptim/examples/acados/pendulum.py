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
    BiorbdModel,
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
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

    bio_model = BiorbdModel(biorbd_model_path)
    nq = bio_model.nb_q
    nqdot = bio_model.nb_qdot

    target = np.zeros((nq + nqdot, 1))
    target[1, 0] = 3.14

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100.0, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=10.0, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1.0, multi_thread=False)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, weight=5000000, key="q", target=target[:nq, :], multi_thread=False
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, weight=500, key="qdot", target=target[nq:, :], multi_thread=False
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model.bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][:, 0] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (nq + nqdot))

    # Define control path constraint
    n_tau = bio_model.nb_tau
    torque_min, torque_max, torque_init = -300, 300, 0
    u_bounds = BoundsList()
    u_bounds.add([torque_min] * n_tau, [torque_max] * n_tau)
    u_bounds[0][n_tau - 1, :] = 0

    u_init = InitialGuessList()
    u_init.add([torque_init] * n_tau)

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        use_sx=use_sx,
        assume_phase_dynamics=True,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization using ACADOS and animates it
    """

    ocp = prepare_ocp(biorbd_model_path="models/pendulum.bioMod", final_time=1, n_shooting=100)

    # --- Solve the program --- #
    solver = Solver.ACADOS()
    solver.set_maximum_iterations(500)
    sol = ocp.solve(solver=solver)

    # --- Show results --- #
    sol.print_cost()
    sol.graphs()
    sol.animate()


if __name__ == "__main__":
    main()
