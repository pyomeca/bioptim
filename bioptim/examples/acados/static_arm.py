"""
TODO: Cleaning
This is a basic example on how to use muscle driven to perform an optimal reaching task.
The arm must reach a marker while minimizing the muscles activity and the states. The problem is solved using both
ACADOS and Ipopt.
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
    InitialGuess,
    Solver,
    InterpolationType,
)


def prepare_ocp(biorbd_model_path, final_time, n_shooting, x_warm=None, use_sx=False, n_threads=1):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -50, 50, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=10, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=10, multi_thread=False)
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, weight=100000, first_marker="target", second_marker="COM_hand"
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_torque=True)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0][:, 0] = (1.0, 1.0, 0, 0)

    # Initial guess
    if x_warm is None:
        x_init = InitialGuess([1.57] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot())
    else:
        x_init = InitialGuess(x_warm, interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
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
        use_sx=use_sx,
        n_threads=n_threads,
    )


def main():
    # Options
    warm_start_ipopt_from_acados_solution = False

    # --- Solve the program using ACADOS --- #
    ocp_acados = prepare_ocp(biorbd_model_path="models/arm26.bioMod", final_time=1, n_shooting=100, use_sx=True)

    solver_acados = Solver.ACADOS()
    solver_acados.set_convergence_tolerance(1e-3)

    sol_acados = ocp_acados.solve(solver=solver_acados)

    # --- Solve the program using IPOPT --- #
    x_warm = sol_acados["qqdot"] if warm_start_ipopt_from_acados_solution else None
    ocp_ipopt = prepare_ocp(
        biorbd_model_path="models/arm26.bioMod",
        final_time=1,
        x_warm=x_warm,
        n_shooting=51,
        use_sx=False,
        n_threads=6,
    )

    solver_ipopt = Solver.IPOPT()
    solver_ipopt.set_linear_solver("ma57")
    solver_ipopt.set_dual_inf_tol(1e-3)
    solver_ipopt.set_constraint_tolerance(1e-3)
    solver_ipopt.set_convergence_tolerance(1e-3)
    solver_ipopt.set_maximum_iterations(100)
    solver_ipopt.set_hessian_approximation("exact")

    sol_ipopt = ocp_ipopt.solve(solver=solver_ipopt)

    # --- Show results --- #
    print("\n\n")
    print("Results using ACADOS")
    print(f"Final objective: {np.nansum(sol_acados.cost)}")
    sol_acados.print_cost()
    print(f"Time to solve: {sol_acados.real_time_to_optimize}sec")
    print(f"")

    print(
        f"Results using Ipopt{'' if warm_start_ipopt_from_acados_solution else ' not'} "
        f"warm started from ACADOS solution"
    )
    print(f"Final objective : {np.nansum(sol_ipopt.cost)}")
    sol_ipopt.print_cost()
    print(f"Time to solve: {sol_ipopt.real_time_to_optimize}sec")
    print(f"")

    visualizer = sol_acados.animate(show_now=False)
    visualizer.extend(sol_ipopt.animate(show_now=False))

    # Update biorbd-viz by hand so they can be visualized simultaneously
    should_continue = True
    while should_continue:
        for i, b in enumerate(visualizer):
            if b.vtk_window.is_active:
                b.update()
            else:
                should_continue = False


if __name__ == "__main__":
    main()
