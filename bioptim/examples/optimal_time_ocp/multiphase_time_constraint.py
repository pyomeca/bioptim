"""
This example is a trivial multiphase box that must superimpose different markers at beginning and end of each
phase with one of its corner. The time is free for each phase
It is designed to show how one can define a multi-phase ocp problem with free time.
"""

import platform

from bioptim import (
    BiorbdModel,
    Solver,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    Node,
    OdeSolverBase,
)


def prepare_ocp(
    final_time: tuple,
    time_min: tuple,
    time_max: tuple,
    n_shooting: tuple,
    biorbd_model_path: str = "models/cube.bioMod",
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
) -> OptimalControlProgram:
    """
    Prepare the optimal control program. This example can be called as a normal single phase (all list len equals to 1)
    or as a three phases (all list len equals to 3)

    Parameters
    ----------
    final_time: list
        The initial guess for the final time of each phase
    time_min: list
        The minimal time for each phase
    time_max: list
        The maximal time for each phase
    n_shooting: list
        The number of shooting points for each phase
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The multiphase OptimalControlProgram ready to be solved
    """

    # --- Options --- #
    n_phases = len(n_shooting)
    if n_phases != 1 and n_phases != 3:
        raise RuntimeError("Number of phases must be 1 to 3")

    # BioModel path
    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path))

    # Problem parameters
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    if n_phases == 3:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)

    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0, expand=expand)
    if n_phases == 3:
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=1, expand=expand)
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2, expand=expand)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=0)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[0], max_bound=time_max[0], phase=0)
    if n_phases == 3:
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m1", phase=1
        )
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[1], max_bound=time_max[1], phase=1
        )
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=2
        )
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[2], max_bound=time_max[2], phase=2
        )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))  # Phase 0
    if n_phases == 3:
        x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))  # Phase 1
        x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))  # Phase 2

    for bounds in x_bounds:
        for i in [1, 3, 4, 5]:
            bounds[i, [0, -1]] = 0
    x_bounds[0][2, 0] = 0.0
    if n_phases == 3:
        x_bounds[2][2, [0, -1]] = [0.0, 1.57]

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
    if n_phases == 3:
        x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
        x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)
    if n_phases == 3:
        u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)
        u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model[0].nb_tau)
    if n_phases == 3:
        u_init.add([tau_init] * bio_model[0].nb_tau)
        u_init.add([tau_init] * bio_model[0].nb_tau)

    # ------------- #

    return OptimalControlProgram(
        bio_model[:n_phases],
        dynamics,
        n_shooting,
        final_time[:n_phases],
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
        assume_phase_dynamics=True,
    )


def main():
    """
    Run a multiphase problem with free time phases and animate the results
    """

    final_time = (2, 5, 4)
    time_min = (1, 3, 0.1)
    time_max = (2, 4, 0.8)
    ns = (20, 30, 20)
    ocp = prepare_ocp(final_time=final_time, time_min=time_min, time_max=time_max, n_shooting=ns)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    param = sol.parameters
    print(f"The optimized phase time are: {param['time'][0, 0]}s, {param['time'][1, 0]}s and {param['time'][2, 0]}s.")
    sol.animate()


if __name__ == "__main__":
    main()
