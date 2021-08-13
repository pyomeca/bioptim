"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways. Fatigue
is applied on torques.

This simple example is a good place to start investigating bioptim as it describes the most common dynamics out there
(the joint torque driven), it defines an objective function and some boundaries and initial guesses.

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution.
"""

import biorbd_casadi as biorbd

from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    QAndQDotBounds,
    InitialGuess,
    Objective,
    FatigueBounds,
    FatigueInitialGuess,
    FatigueList,
    XiaFatigue,
    XiaTauFatigue,
    ObjectiveFcn,
    VariableType,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
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
    fatigue: list
        The type of fatigue applied on the system

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics

    # Fatigue parameters
    fatigue_dynamics = FatigueList()
    for i in range(n_tau):
        fatigue_dynamics.add(
            XiaTauFatigue(
                XiaFatigue(LD=100, LR=100, F=5, R=10, scale=tau_min),
                XiaFatigue(LD=100, LR=100, F=5, R=10, scale=tau_max),
            ),
            state_only=False,
        )
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, fatigue=fatigue_dynamics)

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14
    x_bounds.concatenate(FatigueBounds(fatigue_dynamics))

    # Initial guess
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    x_init = InitialGuess([0] * (n_q + n_qdot))
    x_init.concatenate(FatigueInitialGuess(fatigue_dynamics))

    # Define control path constraint
    u_bounds = FatigueBounds(fatigue_dynamics, variable_type=VariableType.CONTROLS)
    u_init = FatigueInitialGuess(fatigue_dynamics, variable_type=VariableType.CONTROLS)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        use_sx=True,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=3, n_shooting=100)

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show the results in a bioviz animation --- #
    sol.print()
    sol.animate()


if __name__ == "__main__":
    main()
