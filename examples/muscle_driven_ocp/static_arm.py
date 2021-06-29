"""
This is a basic example on how to use biorbd model driven by muscle to perform an optimal reaching task.
The arms must reach a marker placed upward in front while minimizing the muscles activity

Please note that using show_meshes=True in the animator may be long due to the creation of a huge CasADi graph of the
mesh points.
"""

import biorbd
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    weight: float,
    ode_solver: OdeSolver = OdeSolver.RK4(),
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
    weight: float
        The weight applied to the SUPERIMPOSE_MARKERS final objective function. The bigger this number is, the greater
        the model will try to reach the marker. This is in relation with the other objective functions
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, name="tau")
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL)
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, first_marker="target", second_marker="COM_hand", weight=weight
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_residual_torque=True)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0][:, 0] = (0.07, 1.4, 0, 0)

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([1.57] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot())

    # Define control path constraint
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5
    tau_min, tau_max, tau_init = -1, 1, 0
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min] * biorbd_model.nbGeneralizedTorque() + [muscle_min] * biorbd_model.nbMuscleTotal(),
        [tau_max] * biorbd_model.nbGeneralizedTorque() + [muscle_max] * biorbd_model.nbMuscleTotal(),
    )

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque() + [muscle_init] * biorbd_model.nbMuscleTotal())
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
        ode_solver=ode_solver,
    )


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """

    ocp = prepare_ocp(biorbd_model_path="arm26.bioMod", final_time=3, n_shooting=50, weight=1000)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    sol.animate(show_meshes=True)


if __name__ == "__main__":
    main()
