"""
This example presents how to implement a holonomic constraint in bioptim.
The simulation is three bars that are forced to be coherent with a holonomic constraint. It is then a triple
pendulum simulation.
"""
import numpy as np

from bioptim import (
    BiMappingList,
    BiorbdModelHolonomic,
    BoundsList,
    CostType,
    DynamicsList,
    InitialGuessList,
    ObjectiveFcn,
    ObjectiveList,
    OptimalControlProgram,
    Solver,
)

from bioptim.examples.discrete_mechanics_and_optimal_control.holonomic_constraints import HolonomicConstraintFcn

from bioptim.examples.holonomic_constraints.two_pendulums import custom_configure, custom_dynamic


def prepare_ocp(
    biorbd_model_path: str,
    final_time,
    n_shooting,
    use_sx: bool = False,
) -> (OptimalControlProgram, BiorbdModelHolonomic):
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    use_sx: bool
        If SX should be used instead of MX

    Returns
    -------
    The ocp ready to be solved
    """
    bio_model = BiorbdModelHolonomic(biorbd_model_path)
    # Holonomic constraints
    constraint, constraint_jacobian, constraint_double_derivative = HolonomicConstraintFcn.superimpose_markers(
        bio_model,
        "m1",  # marker names
        "m2",
        index=slice(0, 2),  # only constraint on x and y
        local_frame_index=1,  # seems better in one local frame than in global frame, the constraint deviates less
    )
    bio_model.add_holonomic_constraint(
        constraint=constraint,
        constraint_jacobian=constraint_jacobian,
        constraint_double_derivative=constraint_double_derivative,
    )
    bio_model.set_dependencies(independent_joint_index=[0, 1, 4], dependent_joint_index=[2, 3])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.1, min_bound=0.45, max_bound=1.2)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=custom_dynamic, expand=False)

    # Path constraint
    pose_at_first_node = [np.pi / 2, 0, 0]

    # Boundaries
    mapping = BiMappingList()
    mapping.add("q", to_second=[0, 1, None, None, 2], to_first=[0, 1, 4])
    mapping.add("qdot", to_second=[0, 1, None, None, 2], to_first=[0, 1, 4])
    x_bounds = BoundsList()
    x_bounds["u"] = bio_model.bounds_from_ranges("q", mapping=mapping)
    x_bounds["udot"] = bio_model.bounds_from_ranges("qdot", mapping=mapping)
    x_bounds["u"][:, 0] = pose_at_first_node
    x_bounds["udot"][:, 0] = [0] * 3
    x_bounds["u"][:, -1] = [-np.pi / 2, 0, -np.pi / 2]
    x_bounds["udot"][:, -1] = [0] * 3

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("u", pose_at_first_node)
    x_init.add("udot", [0] * 3)

    # Define control path constraint
    tau_min, tau_max, tau_init = -1000, 1000, 0
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * bio_model.nb_tau, max_bound=[tau_max] * bio_model.nb_tau)
    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * bio_model.nb_tau)

    # ------------- #

    return (
        OptimalControlProgram(
            bio_model=bio_model,
            dynamics=dynamics,
            n_shooting=n_shooting,
            phase_time=final_time,
            x_init=x_init,
            u_init=u_init,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            assume_phase_dynamics=True,
            use_sx=use_sx,
        ),
        bio_model,
    )


def main():
    """
    Solve and animate the solution
    """

    n_shooting = 100
    ocp, bio_model = prepare_ocp(
        biorbd_model_path="models/three_bars.bioMod",
        final_time=1,
        n_shooting=n_shooting,
    )

    ocp.add_plot_penalty(cost_type=CostType.ALL)

    # --- Solve the program --- #
    # show_online_optim not working yet
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_constraint_tolerance(1e-15)  # the more it is, the less the constraint is derived
    sol = ocp.solve()

    q = np.zeros((5, n_shooting + 1))
    for i, ui in enumerate(sol.states["u"].T):
        vi = bio_model.compute_v_from_u_numeric(ui, v_init=np.zeros(2)).toarray()
        qi = bio_model.q_from_u_and_v(ui[:, np.newaxis], vi).toarray().squeeze()
        q[:, i] = qi

    import bioviz

    viz = bioviz.Viz("models/three_bar.bioMod", show_contacts=False)
    viz.load_movement(q)
    viz.exec()


if __name__ == "__main__":
    main()
