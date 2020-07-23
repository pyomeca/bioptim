import biorbd
from time import time
import numpy as np

from biorbd_optim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    QAndQDotBounds,
    InitialConditionsList,
    ShowResult,
    Solver,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, use_SX=False):
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
    x_bounds[0].min[:, 0] = (1.0, 1.0, 0, 0)
    x_bounds[0].max[:, 0] = (1.0, 1.0, 0, 0)

    # Initial guess
    x_init = InitialConditionsList()
    x_init.add([1.57] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot())

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            [tau_min] * biorbd_model.nbGeneralizedTorque() + [muscle_min] * biorbd_model.nbMuscleTotal(),
            [tau_max] * biorbd_model.nbGeneralizedTorque() + [muscle_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    u_init = InitialConditionsList()
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
    )


if __name__ == "__main__":
    ocp_ac = prepare_ocp(biorbd_model_path="arm26.bioMod", final_time=2, number_shooting_points=51, use_SX=True)

    # --- Solve the program --- #
    tic = time()
    sol_ac, sol_obj_ac = ocp_ac.solve(
        solver=Solver.ACADOS,
        show_online_optim=False,
        solver_options={"nlp_solver_tol_comp": 1e-3, "nlp_solver_tol_eq": 1e-3, "nlp_solver_tol_stat": 1e-3,},
        return_objectives=True,
    )
    toc_ac = time() - tic

    ocp_ip = prepare_ocp(biorbd_model_path="arm26.bioMod", final_time=2, number_shooting_points=51, use_SX=False)
    tic = time()
    sol_ip, sol_obj_ip = ocp_ip.solve(
        solver=Solver.IPOPT,
        show_online_optim=False,
        solver_options={
            "tol": 1e-3,
            "dual_inf_tol": 1e-3,
            "constr_viol_tol": 1e-3,
            "compl_inf_tol": 1e-3,
            "linear_solver": "ma57",
            "max_iter": 100,
        },
        return_objectives=True,
    )
    toc_ip = time() - tic
    print(f"Time to solve with ACADOS: {toc_ac}sec")
    print(f"Time to solve with IPOPT: {toc_ip}sec")

    print(f"Final objective value ACADOS : {np.nansum(sol_obj_ac)} \n")
    print(f"Final objective value IPOPT : {np.nansum(sol_obj_ip)} \n")

    analyse_ac = Objective.Analyse(ocp_ac, sol_obj_ac)
    analyse_ac.by_function()
    analyse_ac.by_nodes()

    analyse_ip = Objective.Analyse(ocp_ip, sol_obj_ip)
    analyse_ip.by_function()
    analyse_ip.by_nodes()

    # --- Show results --- #
    result_ac = ShowResult(ocp_ac, sol_ac)
    result_ip = ShowResult(ocp_ip, sol_ip)
    result_ac.graphs()
    result_ip.graphs()
    result_ac.animate()
    result_ip.animate()
