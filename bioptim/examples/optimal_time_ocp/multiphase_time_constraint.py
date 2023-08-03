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
    OdeSolver,
    Node,
    OdeSolverBase,
    BiMapping,
)


def prepare_ocp(
    final_time: tuple,
    time_min: tuple,
    time_max: tuple,
    n_shooting: tuple,
    biorbd_model_path: str = "models/cube.bioMod",
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    assume_phase_dynamics: bool = True,
    with_phase_time_equality: bool = False,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the optimal control program. This example can be called as a normal single phase (all list len equals to 1)
    or as a three phases program (all list len equals to 3)

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
    ode_solver: OdeSolverBase
        The ode solver to use
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node
    with_phase_time_equality: bool
        If the phase time equality should be applied, this is ignored if len(n_shooting) = 1 (instead of 3)
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The multiphase OptimalControlProgram ready to be solved
    """

    # --- Options --- #
    n_phases = len(n_shooting)
    if n_phases != 1 and n_phases != 3:
        raise RuntimeError("Number of phases must be 1 to 3")
    time_phase_mapping = None
    if n_phases and with_phase_time_equality:
        # First and last phase should have the same time
        time_phase_mapping = BiMapping(to_second=[0, 1, 0], to_first=[0, 1])

    # BioModel path
    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path))

    # Problem parameters
    tau_min, tau_max = -100, 100

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    if n_phases == 3:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0, expand=expand_dynamics)
    if n_phases == 3:
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=1, expand=expand_dynamics)
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2, expand=expand_dynamics)

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
    x_bounds.add("q", bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bio_model[0].bounds_from_ranges("qdot"), phase=0)
    if n_phases == 3:
        x_bounds.add("q", bio_model[1].bounds_from_ranges("q"), phase=1)
        x_bounds.add("qdot", bio_model[1].bounds_from_ranges("qdot"), phase=1)
        x_bounds.add("q", bio_model[2].bounds_from_ranges("q"), phase=2)
        x_bounds.add("qdot", bio_model[2].bounds_from_ranges("qdot"), phase=2)

    for bounds in x_bounds:
        bounds["q"][1, [0, -1]] = 0
        bounds["qdot"][:, [0, -1]] = 0
    x_bounds[0]["q"][2, 0] = 0.0
    if n_phases == 3:
        x_bounds[2]["q"][2, [0, -1]] = [0.0, 1.57]

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=0)
    if n_phases == 3:
        u_bounds.add(
            "tau", min_bound=[tau_min] * bio_model[1].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=1
        )
        u_bounds.add(
            "tau", min_bound=[tau_min] * bio_model[2].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=2
        )

    # ------------- #

    return OptimalControlProgram(
        bio_model[:n_phases],
        dynamics,
        n_shooting,
        final_time[:n_phases],
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        time_phase_mapping=time_phase_mapping,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    Run a multiphase problem with free time phases and animate the results
    """

    final_time = (2, 5, 4)
    time_min = (0.7, 3, 0.1)
    time_max = (2, 4, 1)
    ns = (20, 30, 20)
    ocp = prepare_ocp(
        final_time=final_time, time_min=time_min, time_max=time_max, n_shooting=ns, with_phase_time_equality=True
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    time = [sol.parameters["time"][i, 0] for i in ocp.time_phase_mapping.to_second.map_idx]
    print(f"The optimized phase time are: {time[0]}s, {time[1]}s and {time[2]}s.")
    sol.animate()


if __name__ == "__main__":
    main()
