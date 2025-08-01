"""
This is a basic example on how to use biorbd model driven by muscle to perform an optimal reaching task with a
contact dynamics.
The arms must reach a marker placed upward in front while minimizing the muscles activity

Please note that using show_meshes=True in the animator may be long due to the creation of a huge CasADi graph of the
mesh points.
"""

import platform

from bioptim import (
    MusclesBiorbdModel,
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsOptions,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    ContactType,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    weight: float,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
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
    ode_solver: OdeSolverBase
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = MusclesBiorbdModel(
        biorbd_model_path, with_residual_torque=True, contact_types=[ContactType.RIGID_EXPLICIT]
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles")
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, first_marker="target", second_marker="COM_hand", weight=weight
    )

    # Dynamics
    dynamics = DynamicsOptions(
        ode_solver=ode_solver,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["q"][:, 0] = (0, 0.07, 1.4)
    x_bounds["qdot"][:, 0] = (0, 0, 0)

    # Initial guess
    x_init = InitialGuessList()
    x_init["q"] = [1.57] * bio_model.nb_q

    # Define control path constraint
    muscle_min, muscle_max, muscle_init = 0.0, 1.0, 0.5
    tau_min, tau_max, tau_init = -1.0, 1.0, 0.0
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau
    u_bounds["muscles"] = (
        [muscle_min] * bio_model.nb_muscles,
        [muscle_max] * bio_model.nb_muscles,
    )

    u_init = InitialGuessList()
    # ------------- #

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
    )


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """

    ocp = prepare_ocp(biorbd_model_path="models/arm26_with_contact.bioMod", final_time=1, n_shooting=30, weight=1000)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    sol.print_cost()
    sol.animate()


if __name__ == "__main__":
    main()
