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
    TorqueBiorbdModel,
    Node,
    OptimalControlProgram,
    Dynamics,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    PhaseDynamics,
)


def prepare_ocp(
    biorbd_model_path: str = "models/cubeSym.bioMod",
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        Path to the bioMod
    ode_solver: OdeSolverBase
        The ode solver to use
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = TorqueBiorbdModel(biorbd_model_path)

    # Problem parameters
    n_shooting = 30
    final_time = 2
    tau_min, tau_max = -100, 100

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, index=[0, 1, 3], key="tau", weight=100)

    # Dynamics
    dynamics = Dynamics(ode_solver=ode_solver, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")
    constraints.add(
        ConstraintFcn.PROPORTIONAL_CONTROL, key="tau", node=Node.ALL_SHOOTING, first_dof=3, second_dof=4, coef=-1
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["q"][2, :] = 0  # Third dof is set to zero
    x_bounds["qdot"][:, [0, -1]] = 0  # Velocity is 0 at start and end

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * bio_model.nb_q, [tau_max] * bio_model.nb_q
    u_bounds["tau"][2, :] = 0  # Third dof is left uncontrolled

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
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
