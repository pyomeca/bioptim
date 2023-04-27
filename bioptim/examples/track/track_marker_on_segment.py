"""
This example is a trivial example where a stick must keep a corner of a box in line for the whole duration of the
movement. The initial and final position of the box are dictated, the rest is fully optimized. It is designed
to show how one can use the tracking function to track a marker with a body segment
"""

import biorbd_casadi as biorbd
from bioptim import (
    BiorbdModel,
    Node,
    Axis,
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
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    initialize_near_solution: bool,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    constr: bool = True,
    use_sx: bool = False,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod file
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    initialize_near_solution: bool
        If the initial guess should be almost the solution (this is merely to reduce the time of the tests)
    ode_solver: OdeSolver
        The ode solver to use
    constr: bool
        If the constraint should be applied (this is merely to reduce the time of the tests)
    use_sx: bool
        If SX CasADi variables should be used

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)

    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

    # Constraints
    if constr:
        constraints = ConstraintList()
        constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m4")
        constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m5")
        constraints.add(
            ConstraintFcn.TRACK_MARKER_WITH_SEGMENT_AXIS, node=Node.ALL, marker="m1", segment="seg_rt", axis=Axis.X
        )
    else:
        constraints = ConstraintList()

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model.bounds_from_ranges(["q", "qdot"]))

    for i in range(1, 8):
        if i != 3:
            x_bounds[0][i, [0, -1]] = 0
    x_bounds[0][2, -1] = 1.57

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (bio_model.nb_q + bio_model.nb_qdot))
    if initialize_near_solution:
        for i in range(2):
            x_init[0].init[i] = 1.5
        for i in range(4, 6):
            x_init[0].init[i] = 0.7
        for i in range(6, 8):
            x_init[0].init[i] = 0.6

    # Define control path constraint
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau)

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model.nb_tau)

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
        use_sx=use_sx,
        assume_phase_dynamics=True,
    )


def main():
    """
    Prepares, solves and animate the program
    """

    ocp = prepare_ocp(
        biorbd_model_path="models/cube_and_line.bioMod",
        n_shooting=30,
        final_time=2,
        initialize_near_solution=True,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
