"""
TODO: Cleaning
This is a basic example on how to use muscle driven to perform an optimal reaching task.
The arm must reach a marker while minimizing the muscles activity and the states. The problem is solved using both
ACADOS and Ipotpt.
"""

import numpy as np
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    ObjectiveFcn,
    ObjectiveList,
    Bounds,
    InitialGuess,
    OdeSolver,
    Solver,
)


def prepare_ocp(biorbd_model_path, n_shooting, tf, ode_solver=OdeSolver.RK4(), use_sx=True):
    # BioModel path
    bio_model = BiorbdModel(biorbd_model_path)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_init = InitialGuess([0] * (bio_model.nb_q + bio_model.nb_qdot))

    # Define control path constraint
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau)
    u_init = InitialGuess([tau_init] * bio_model.nb_tau)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        tf,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        ode_solver=ode_solver,
        use_sx=use_sx,
        assume_phase_dynamics=True,
    )


def main():
    model_path = "models/cube.bioMod"
    ns = 30
    tf = 2
    ocp = prepare_ocp(biorbd_model_path=model_path, n_shooting=ns, tf=tf)

    # --- Add objective functions --- #
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="q",
        weight=1000,
        index=[0, 1],
        target=np.array([[1.0, 2.0]]).T,
        multi_thread=False,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="q",
        weight=10000,
        index=[2],
        target=np.array([[3.0]]),
        multi_thread=False,
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, multi_thread=False)
    ocp.update_objectives(objective_functions)

    # --- Solve the program --- #
    solver = Solver.ACADOS()
    sol = ocp.solve(solver)
    sol.graphs()

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="q",
        weight=1,
        index=[0, 1],
        target=np.array([[1.0, 2.0]]).T,
        multi_thread=False,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="q",
        weight=10000,
        index=[2],
        target=np.array([[3.0]]),
        multi_thread=False,
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, multi_thread=False)
    ocp.update_objectives(objective_functions)

    solver.set_nlp_solver_tol_stat(1e-2)
    sol = ocp.solve(solver)

    # --- Show results --- #
    sol.graphs()
    sol.animate()


if __name__ == "__main__":
    main()
