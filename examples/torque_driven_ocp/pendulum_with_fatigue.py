"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This simple example is a good place to start investigating bioptim as it describes the most common dynamics out there
(the joint torque driven), it defines an objective function and some boundaries and initial guesses

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution
"""

from bioptim.misc.enums import Fatigue

import biorbd_casadi as biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    XiaFatigueDynamicsList,
    XiaFatigueStateBounds,
    XiaFatigueStateInitialGuess,
    XiaFatigueControlsInitialGuess,
    XiaFatigueControlsBounds,
    XiaTorqueFatigue,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    fatigue: list = [],  # TODO: Do not use mutables as default arguments
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
        A list of fatigue elements

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)
    use_fatigue = True if Fatigue.TAU in fatigue or Fatigue.TAU_STATE_ONLY in fatigue else False

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, fatigue=fatigue, expand=False)

    # Path constraint
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    if use_fatigue:
        x_bounds.concatenate(XiaFatigueStateBounds(biorbd_model, has_torque=True))

    # Initial guess
    tau_min, tau_max, tau_init = -100, 100, 0

    x_init = InitialGuess([0] * (n_q + n_qdot))
    u_init = InitialGuess([tau_init] * n_tau)
    if use_fatigue:
        x_init.concatenate(
            XiaFatigueStateInitialGuess(biorbd_model, has_torque=True, tau_init=tau_init, tau_max=tau_max)
        )
        u_init = XiaFatigueControlsInitialGuess(biorbd_model, torque=u_init)

    # Define control path constraint
    u_min = [tau_min] * n_tau
    u_max = [tau_max] * n_tau
    u_bounds = Bounds(u_min, u_max)
    if use_fatigue:
        u_bounds = XiaFatigueControlsBounds(biorbd_model, torque=u_bounds)

    # Define fatigue parameters
    fatigue_dynamics = XiaFatigueDynamicsList()

    fatigue_dynamics.add_torque(
        XiaTorqueFatigue(LD=100, LR=100, F=0.9, R=0.01, tau_max=-100),
        XiaTorqueFatigue(LD=100, LR=100, F=0.9, R=0.01, tau_max=100), index=0
    )
    fatigue_dynamics.add_torque(
        XiaTorqueFatigue(LD=100, LR=100, F=0.9, R=0.01, tau_max=-100),
        XiaTorqueFatigue(LD=100, LR=100, F=0.9, R=0.01, tau_max=100), index=1
    )

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
        fatigue_dynamics=fatigue_dynamics,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(
        biorbd_model_path="../getting_started/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        fatigue=[Fatigue.TAU_STATE_ONLY]
    )

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=True)

    # --- Solve the ocp --- #
    sol = ocp.solve(show_online_optim=True)

    # Save results with stand alone
    # file_path = "results/with_fatigue_tau_state_only.bo"
    # ocp.save(sol, file_path, True)

    # --- Show the results in a bioviz animation --- #
    sol.print()
    sol.animate()


if __name__ == "__main__":
    main()