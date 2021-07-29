"""
This is a basic example on how to use biorbd model driven by muscle to perform an optimal reaching task.
Fatigue is applied on muscles. The arms must reach a marker placed upward in front while minimizing
the muscles activity and fatigue.
Please note that using show_meshes=True in the animator may be long due to the creation of a huge CasADi graph of the
mesh points.
"""

import biorbd_casadi as biorbd

from bioptim.misc.enums import Fatigue
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    Dynamics,
    DynamicsFcn,
    QAndQDotBounds,
    InitialGuess,
    OdeSolver,
    Bounds,
    Constraint,
    ConstraintFcn,
    XiaFatigueDynamicsList,
    XiaFatigueStateInitialGuess,
    XiaFatigueStateBounds,
    XiaFatigueControlsBounds,
    XiaFatigueControlsInitialGuess,
    Node,
    Axis,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    with_residual_torque: bool = False,
    fatigue: list = None,
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
    ode_solver: OdeSolver
        The ode solver to use
    with_residual_torque: bool
        True if we use residual torque
    fatigue: list
        The type of fatigue applied on the system
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    n_muscles = biorbd_model.nbMuscleTotal()
    muscle_min, muscle_max, muscle_init = 0, 1, 0.3

    tau_min, tau_max, tau_init = -1, 1, 0

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.MUSCLE_DRIVEN, expand=False, fatigue=fatigue, with_residual_torque=with_residual_torque
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1)
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, first_marker="target", second_marker="COM_hand", weight=0.01
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="muscles_mf", weight=10000)  # Minimize fatigue

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
    x_bounds.concatenate(
        XiaFatigueStateBounds(biorbd_model, has_muscles=True, has_torque=False)
    )

    x_init = InitialGuess([1.57] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot())
    x_init.concatenate(
        XiaFatigueStateInitialGuess(
            biorbd_model,
            has_muscles=True,
            has_torque=False,
            muscle_init=muscle_init,
            tau_init=tau_init,
            tau_max=tau_max,
        )
    )

    # Define control path constraint
    muscle_bounds = Bounds([muscle_min] * n_muscles, [muscle_max] * n_muscles)
    muscle_init = InitialGuess([muscle_init] * n_muscles)
    u_bounds = XiaFatigueControlsBounds(biorbd_model, muscles=muscle_bounds)
    u_init = XiaFatigueControlsInitialGuess(biorbd_model, muscles=muscle_init)

    # Define fatigue parameters for each muscle and residual torque
    fatigue_dynamics = XiaFatigueDynamicsList()
    for _ in range(n_muscles):
        fatigue_dynamics.add_muscle(LD=10, LR=10, F=0.01, R=0.002)

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
        constraint,
        ode_solver=ode_solver,
        fatigue_dynamics=fatigue_dynamics,
        use_sx=False,
        n_threads=8,
    )


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """

    ocp = prepare_ocp(
        biorbd_model_path="arm26_constant.bioMod",
        final_time=3,
        n_shooting=50,
        with_residual_torque=False,
        fatigue=[Fatigue.MUSCLES_STATE_ONLY]
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True, solver_options={"hessian_approximation": "exact"})
    sol.print()

    # --- Show results --- #
    sol.animate(show_meshes=True)


if __name__ == "__main__":
    main()
