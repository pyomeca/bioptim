"""
This trivial example has two rodes and must superimpose a marker on one rod at the beginning and another marker on the
same rod at the end, while keeping the degrees of freedom opposed. It does this by imposing the symmetry as a
proportional constraint (equals to -1).

The proportional constraint simply creates a constraint such that: state[i] = coef * state[j], where the coef is the
proportional constraint.

The difference between symmetry_by_mapping and symmetry_by_constraint is that one (mapping) removes the degree of
freedom from the solver, while the other (constraints) imposes a proportional constraint (equals to -1) so they
are opposed.
Please note that even though removing a degree of freedom seems a good idea, it is unclear if it is actually faster when
solving with IPOPT.
"""

import platform

from bioptim import (
    BiorbdModel,
    Node,
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
    OdeSolverBase,
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str = "models/cubeSym.bioMod", ode_solver: OdeSolverBase = OdeSolver.RK4(), assume_phase_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        Path to the bioMod
    ode_solver: OdeSolver
        The ode solver to use
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Problem parameters
    n_shooting = 30
    final_time = 2
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, index=[0, 1, 3], key="tau", weight=100)

    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")
    constraints.add(
        ConstraintFcn.PROPORTIONAL_CONTROL, key="tau", node=Node.ALL_SHOOTING, first_dof=3, second_dof=4, coef=-1
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model.bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][2, :] = 0  # Third dof is set to zero
    x_bounds[0][bio_model.nb_q :, [0, -1]] = 0  # Velocity is 0 at start and end

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (bio_model.nb_q + bio_model.nb_qdot))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * bio_model.nb_q, [tau_max] * bio_model.nb_q)
    u_bounds[0][2, :] = 0  # Third dof is left uncontrolled

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model.nb_q)

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
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    Solves an ocp where the symmetry is enforced by constraints, and animates it
    """

    ocp = prepare_ocp()

    # Objective and constraints plots
    ocp.add_plot_penalty()

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
