import biorbd
from time import time
import numpy as np

from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    InitialGuessOption,
    ShowResult,
    Solver,
    InterpolationType,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, x_warm=None, use_SX=False, nb_threads=1):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -50, 50, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=10)
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=10)
    objective_functions.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10)
    objective_functions.add(Objective.Mayer.ALIGN_MARKERS, weight=100000, first_marker_idx=0, second_marker_idx=1)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0][:, 0] = (1.0, 1.0, 0, 0)

    # Initial guess
    if x_warm is None:
        x_init = InitialGuessOption([1.57] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot())
    else:
        x_init = InitialGuessOption(x_warm, interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            [tau_min] * biorbd_model.nbGeneralizedTorque() + [muscle_min] * biorbd_model.nbMuscleTotal(),
            [tau_max] * biorbd_model.nbGeneralizedTorque() + [muscle_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque() + [muscle_init] * biorbd_model.nbMuscleTotal())
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        use_SX=use_SX,
        nb_threads=nb_threads,
    )


if __name__ == "__main__":
    # Options
    warm_start_ipopt_from_acados_solution = False

    # --- Solve the program using ACADOS --- #
    ocp_acados = prepare_ocp(biorbd_model_path="arm26.bioMod", final_time=2, number_shooting_points=51, use_SX=True)

    tic = time()
    sol_acados, sol_obj_acados = ocp_acados.solve(
        solver=Solver.ACADOS,
        show_online_optim=False,
        solver_options={
            "nlp_solver_tol_comp": 1e-3,
            "nlp_solver_tol_eq": 1e-3,
            "nlp_solver_tol_stat": 1e-3,
        },
        return_objectives=True,
    )
    toc_acados = time() - tic

    # --- Solve the program using IPOPT --- #
    x_warm = sol_acados["qqdot"] if warm_start_ipopt_from_acados_solution else None
    ocp_ipopt = prepare_ocp(
        biorbd_model_path="arm26.bioMod",
        final_time=2,
        x_warm=x_warm,
        number_shooting_points=51,
        use_SX=False,
        nb_threads=6,
    )

    tic = time()
    sol_ipopt, sol_obj_ipopt = ocp_ipopt.solve(
        solver=Solver.IPOPT,
        show_online_optim=False,
        solver_options={
            "tol": 1e-3,
            "dual_inf_tol": 1e-3,
            "constr_viol_tol": 1e-3,
            "compl_inf_tol": 1e-3,
            "linear_solver": "ma57",
            "max_iter": 100,
            "hessian_approximation": "exact",
        },
        return_objectives=True,
    )
    toc_ipopt = time() - tic

    # --- Show results --- #
    print("\n\n")
    print("Results using ACADOS")
    print(f"Final objective: {np.nansum(sol_obj_acados)}")
    analyse_acados = Objective.Printer(ocp_acados, sol_obj_acados)
    analyse_acados.by_function()
    print(f"Time to solve: {sol_acados['time_tot']}sec")
    print(f"")

    print(
        f"Results using Ipopt{'' if warm_start_ipopt_from_acados_solution else ' not'} warm started from ACADOS solution"
    )
    print(f"Final objective : {np.nansum(sol_obj_ipopt)}")
    analyse_ipopt = Objective.Printer(ocp_ipopt, sol_obj_ipopt)
    analyse_ipopt.by_function()
    print(f"Time to solve: {sol_ipopt['time_tot']}sec")
    print(f"")

    result_acados = ShowResult(ocp_acados, sol_acados)
    result_ipopt = ShowResult(ocp_ipopt, sol_ipopt)
    visualizer = result_acados.animate(show_now=False)
    visualizer.extend(result_ipopt.animate(show_now=False))

    # Update biorbd-viz by hand so they can be visualized simultaneously
    should_continue = True
    while should_continue:
        for i, b in enumerate(visualizer):
            if b.vtk_window.is_active:
                b.update()
            else:
                should_continue = False
