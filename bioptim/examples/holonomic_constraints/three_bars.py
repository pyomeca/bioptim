from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsFcn,
    DynamicsList,
    ObjectiveFcn,
    ObjectiveList,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    Solver,
    CostType,
    BiMappingList,
)
import numpy as np
from ocp_example_2 import generate_close_loop_constraint, custom_configure, custom_dynamic
from biorbd_model_holonomic import BiorbdModelCustomHolonomic
from graphs import constraints_graphs


def prepare_ocp(
    biorbd_model_path: str, phase_time, n_shooting, ode_solver: OdeSolver = OdeSolver.RK4()
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    ode_solver: OdeSolver
        The type of ode solver used

    Returns
    -------
    The ocp ready to be solved
    """

    # --- Options --- #
    bio_model = BiorbdModelCustomHolonomic(biorbd_model_path[0])
    # made up constraints
    constraint, constraint_jacobian, constraint_double_derivative = generate_close_loop_constraint(
        bio_model,
        "m1",  # marker names
        "m2",
        index=slice(0, 2),  # only constraint on x and y
        local_frame_index=1,  # seems better in one local frame than in global frame, the constraint deviates less
    )
    #
    bio_model.add_holonomic_constraint(
        constraint=constraint,
        constraint_jacobian=constraint_jacobian,
        constraint_double_derivative=constraint_double_derivative,
    )

    bio_model.set_dependencies(independent_joint_index=[0, 1, 4], dependent_joint_index=[2, 3])

    # Problem parameters
    tau_min, tau_max, tau_init = -1000, 1000, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.1, min_bound=0.45, max_bound=1.2)

    # Dynamics
    dynamics = DynamicsList()
    # made up dynamics
    dynamics.add(custom_configure, dynamic_function=custom_dynamic, expand=False)

    # Constraints
    constraints = ConstraintList()

    # Path constraint
    pose_at_first_node = [np.pi / 2, 0, 0]

    mapping = BiMappingList()
    mapping.add("q", [0, 1, None, None, 2], [0, 1, 4])
    mapping.add("qdot", [0, 1, None, None, 2], [0, 1, 4])

    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model.bounds_from_ranges(["q", "qdot"], mapping=mapping))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * 3
    x_bounds[0][:, -1] = [-np.pi / 2, 0, -np.pi / 2] + [0] * 3

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * 3)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau)

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model.nb_tau)

    # ------------- #

    return (
        OptimalControlProgram(
            bio_model=bio_model,
            dynamics=dynamics,
            n_shooting=n_shooting,
            phase_time=phase_time,
            x_init=x_init,
            u_init=u_init,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            constraints=constraints,
            ode_solver=ode_solver,
            n_threads=8,
            assume_phase_dynamics=True,
            use_sx=False,
        ),
        bio_model,
    )


def main():
    """
    Solve and animate the solution
    """

    ocp, biomodel = prepare_ocp(
        biorbd_model_path=("models/three_bar.bioMod",),
        phase_time=1,
        n_shooting=100,
    )

    ocp.add_plot_penalty(cost_type=CostType.ALL)

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_constraint_tolerance(1e-15)  # the more it is, the less the constraint is derived
    sol = ocp.solve()

    # sol.graphs(show_bounds=True)
    q = np.zeros((5, 101))
    for i, ui in enumerate(sol.states["u"].T):
        vi = biomodel.compute_v_from_u_numeric(ui, v_init=np.zeros(2)).toarray()
        qi = biomodel.q_from_u_and_v(ui[:, np.newaxis], vi).toarray().squeeze()
        q[:, i] = qi

    import bioviz

    viz = bioviz.Viz("models/three_bar.bioMod", show_contacts=False)
    viz.load_movement(q)
    viz.exec()


if __name__ == "__main__":
    main()
