"""
This example presents how to implement a holonomic constraint in bioptim.
The simulation is three bars that are forced to be coherent with a holonomic constraint. It is then a triple
pendulum simulation.
"""
import numpy as np

import bioviz

from bioptim import (
    BiMappingList,
    BoundsList,
    CostType,
    DynamicsFcn,
    DynamicsList,
    HolonomicBiorbdModel,
    HolonomicConstraintFcn,
    InitialGuessList,
    ObjectiveFcn,
    ObjectiveList,
    OptimalControlProgram,
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time,
    n_shooting,
) -> (OptimalControlProgram, HolonomicBiorbdModel):
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

    Returns
    -------
    The ocp ready to be solved
    """
    bio_model = HolonomicBiorbdModel(biorbd_model_path)
    # Create a holonomic constraint to create a triple pendulum from a double pendulum and a simple pendulum
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
    # The rotations (joint 0, 1 and 4) are independent. The translations (joint 2 and 3) are constrained by the
    # holonomic constraint
    bio_model.set_dependencies(independent_joint_index=[0, 1, 4], dependent_joint_index=[2, 3])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.1, min_bound=0.45, max_bound=1.2)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.HOLONOMIC_TORQUE_DRIVEN, expand=False)

    # Path constraint
    pose_at_first_node = [np.pi / 2, 0, 0]

    # Boundaries
    mapping = BiMappingList()
    # The rotations (joint 0, 1 and 4) are independent. The translations (joint 2 and 3) are constrained by the
    # holonomic constraint, so we need to map the states and controls to only compute the dynamics of the independent
    # joints
    # The dynamics of the dependent joints will be computed from the holonomic constraint
    mapping.add("q", to_second=[0, 1, None, None, 2], to_first=[0, 1, 4])
    mapping.add("qdot", to_second=[0, 1, None, None, 2], to_first=[0, 1, 4])
    x_bounds = BoundsList()
    # q_u and qdot_u are the states of the independent joints
    x_bounds["q_u"] = bio_model.bounds_from_ranges("q", mapping=mapping)
    x_bounds["qdot_u"] = bio_model.bounds_from_ranges("qdot", mapping=mapping)
    x_bounds["q_u"][:, 0] = pose_at_first_node
    x_bounds["qdot_u"][:, 0] = [0] * 3
    x_bounds["q_u"][:, -1] = [-np.pi / 2, 0, -np.pi / 2]
    x_bounds["qdot_u"][:, -1] = [0] * 3

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q_u", pose_at_first_node)
    x_init.add("qdot_u", [0] * 3)

    # Define control path constraint
    variable_bimapping = BiMappingList()
    tau_min, tau_max, tau_init = -1000, 1000, 0
    # Only the rotations are controlled
    variable_bimapping.add("tau", to_second=[0, 1, None, None, 2], to_first=[0, 1, 4])
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * 3, max_bound=[tau_max] * 3)
    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * 3)

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
            variable_mappings=variable_bimapping,
        ),
        bio_model,
    )


def main():
    """
    Solve and animate the solution
    """

    n_shooting = 10
    ocp, bio_model = prepare_ocp(
        biorbd_model_path="models/three_bars.bioMod",
        final_time=1,
        n_shooting=n_shooting,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT())

    q = np.zeros((5, n_shooting + 1))
    for i, ui in enumerate(sol.states["q_u"].T):
        vi = bio_model.compute_q_v_numeric(ui, q_v_init=np.zeros(2)).toarray()
        qi = bio_model.state_from_partition(ui[:, np.newaxis], vi).toarray().squeeze()
        q[:, i] = qi

    viz = bioviz.Viz("models/three_bar.bioMod", show_contacts=False)
    viz.load_movement(q)
    viz.exec()


if __name__ == "__main__":
    main()