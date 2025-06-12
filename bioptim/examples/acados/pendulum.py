"""
TODO: Please confirm this example
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This simple example is a good place to start investigating bioptim using ACADOS as it describes the most common
dynamics out there (the joint torque driven), it defines an objective function and some boundaries and initial guesses
"""

import numpy as np
from bioptim import (
    TorqueBiorbdModel,
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    Dynamics,
    BoundsList,
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    use_sx: bool = True,
    expand_dynamics: bool = True,
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

    bio_model = TorqueBiorbdModel(biorbd_model_path)
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
    dynamics = Dynamics(expand_dynamics=expand_dynamics)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = 0

    # Define control path constraint
    n_tau = bio_model.nb_tau
    torque_min, torque_max = -300, 300
    u_bounds = BoundsList()
    u_bounds["tau"] = [torque_min] * n_tau, [torque_max] * n_tau
    u_bounds["tau"][-1, :] = 0

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        use_sx=use_sx,
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
