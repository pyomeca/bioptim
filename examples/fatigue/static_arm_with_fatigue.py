"""
This is a basic example on how to use biorbd model driven by muscle to perform an optimal reaching task.
Fatigue is applied on muscles. The arms must reach a marker placed upward in front while minimizing
the muscles activity and fatigue.
Please note that using show_meshes=True in the animator may be long due to the creation of a huge CasADi graph of the
mesh points.
"""

import biorbd_casadi as biorbd

from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    Dynamics,
    DynamicsFcn,
    QAndQDotBounds,
    InitialGuess,
    OdeSolver,
    Constraint,
    ConstraintFcn,
    FatigueList,
    FatigueBounds,
    FatigueInitialGuess,
    Bounds,
    XiaFatigue,
    XiaTauFatigue,
    MichaudFatigue,
    MichaudTauFatigue,
    Node,
    Axis,
    VariableType,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    fatigue_type: str,
    ode_solver: OdeSolver = OdeSolver.COLLOCATION(),
    torque_level: int = 0,
) -> OptimalControlProgram:
    """
    Prepare the ocp
    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    fatigue_type: str
        The type of dynamics to apply ("xia" or "michaud")
    ode_solver: OdeSolver
        The ode solver to use
    torque_level: int
        0 no residual torque, 1 with residual torque, 2 with fatigable residual torque
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    n_tau = biorbd_model.nbGeneralizedTorque()
    n_muscles = biorbd_model.nbMuscleTotal()
    tau_min, tau_max = -10, 10

    # Define fatigue parameters for each muscle and residual torque
    fatigue_dynamics = FatigueList()
    for i in range(n_muscles):
        if fatigue_type == "xia":
            fatigue_dynamics.add(XiaFatigue(LD=10, LR=10, F=0.01, R=0.002), state_only=False)
        elif fatigue_type == "michaud":
            fatigue_dynamics.add(MichaudFatigue(LD=10, LR=10, F=0.01, R=0.002, fatigue_threshold=0.15, L=0.07), state_only=False)
        else:
            raise ValueError("fatigue_type not implemented")
    if torque_level >= 2:
        for i in range(n_tau):
            if fatigue_type == "xia":
                fatigue_dynamics.add(
                    XiaTauFatigue(
                        XiaFatigue(LD=10, LR=10, F=5, R=10, scale=tau_min),
                        XiaFatigue(LD=10, LR=10, F=5, R=10, scale=tau_max),
                    ),
                    state_only=False,
                )
            elif fatigue_type == "michaud":
                fatigue_dynamics.add(
                    MichaudTauFatigue(
                        MichaudFatigue(LD=10, LR=10, F=5, R=10, fatigue_threshold=0.15, L=0.07, scale=tau_min),
                        MichaudFatigue(LD=10, LR=10, F=5, R=10, fatigue_threshold=0.15, L=0.07, scale=tau_max),
                    ),
                    state_only=False,
                )
            else:
                raise ValueError("fatigue_type not implemented")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.MUSCLE_DRIVEN, expand=False, fatigue=fatigue_dynamics, with_torque=torque_level > 0)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=100)
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, first_marker="target", second_marker="COM_hand", weight=0.01
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE, key="muscles", weight=1000)

    # Constraint
    constraint = Constraint(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        first_marker="target",
        second_marker="COM_hand",
        node=Node.END,
        axes=[Axis.X, Axis.Y],
    )

    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, 0] = (0.07, 1.4, 0, 0)
    x_bounds.concatenate(FatigueBounds(fatigue_dynamics))

    x_init = InitialGuess([1.57] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot())
    x_init.concatenate(FatigueInitialGuess(fatigue_dynamics))

    # Define control path constraint
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau) if torque_level == 1 else Bounds()
    u_bounds.concatenate(FatigueBounds(fatigue_dynamics, variable_type=VariableType.CONTROLS))
    u_init = InitialGuess([0] * n_tau) if torque_level == 1 else InitialGuess()
    u_init.concatenate(FatigueInitialGuess(fatigue_dynamics, variable_type=VariableType.CONTROLS))

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
        constraint,
        ode_solver=ode_solver,
        use_sx=False,
        n_threads=8,
    )


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """

    ocp = prepare_ocp(
        biorbd_model_path="models/arm26_constant.bioMod",
        final_time=0.8,
        n_shooting=50,
        fatigue_type="michaud",
        torque_level=1,
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True, solver_options={"hessian_approximation": "exact"})
    sol.print()

    # --- Show results --- #
    sol.animate(show_meshes=True)


if __name__ == "__main__":
    main()
