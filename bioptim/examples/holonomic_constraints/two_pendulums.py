"""
This example presents how to implement a holonomic constraint in bioptim.
The simulation is two single pendulum that are forced to be coherent with a holonomic constraint. It is then a double
pendulum simulation.
"""
import matplotlib.pyplot as plt
import numpy as np

from casadi import MX, Function

from bioptim import (
    BiMappingList,
    BoundsList,
    ConstraintList,
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


def compute_all_states(sol, bio_model: HolonomicBiorbdModel):
    """
    Compute all the states from the solution of the optimal control program

    Parameters
    ----------
    bio_model: HolonomicBiorbdModel
        The biorbd model
    sol:
        The solution of the optimal control program

    Returns
    -------

    """
    n = sol.states["q_u"].shape[1]

    q = np.zeros((bio_model.nb_q, n))
    qdot = np.zeros((bio_model.nb_q, n))
    qddot = np.zeros((bio_model.nb_q, n))
    lambdas = np.zeros((bio_model.nb_dependent_joints, n))
    tau = np.zeros((bio_model.nb_tau, n))

    for i, independent_joint_index in enumerate(bio_model.independent_joint_index):
        tau[independent_joint_index] = sol.controls["tau"][i, :]
    for i, dependent_joint_index in enumerate(bio_model.dependent_joint_index):
        tau[dependent_joint_index] = sol.controls["tau"][i, :]

    # Partitioned forward dynamics
    q_u_sym = MX.sym("q_u_sym", bio_model.nb_independent_joints, 1)
    qdot_u_sym = MX.sym("qdot_u_sym", bio_model.nb_independent_joints, 1)
    tau_sym = MX.sym("tau_sym", bio_model.nb_tau, 1)
    partitioned_forward_dynamics_func = Function(
        "partitioned_forward_dynamics",
        [q_u_sym, qdot_u_sym, tau_sym],
        [bio_model.partitioned_forward_dynamics(q_u_sym, qdot_u_sym, tau_sym)],
    )
    # Lagrangian multipliers
    q_sym = MX.sym("q_sym", bio_model.nb_q, 1)
    qdot_sym = MX.sym("qdot_sym", bio_model.nb_q, 1)
    qddot_sym = MX.sym("qddot_sym", bio_model.nb_q, 1)
    compute_lambdas_func = Function(
        "compute_the_lagrangian_multipliers",
        [q_sym, qdot_sym, qddot_sym, tau_sym],
        [bio_model.compute_the_lagrangian_multipliers(q_sym, qdot_sym, qddot_sym, tau_sym)],
    )

    for i in range(n):
        q_v_i = bio_model.compute_q_v_numeric(sol.states["q_u"][:, i]).toarray()
        q[:, i] = bio_model.state_from_partition(sol.states["q_u"][:, i][:, np.newaxis], q_v_i).toarray().squeeze()
        qdot[:, i] = bio_model.compute_qdot(q[:, i], sol.states["qdot_u"][:, i]).toarray().squeeze()
        qddot_u_i = (
            partitioned_forward_dynamics_func(
                sol.states["q_u"][:, i],
                sol.states["qdot_u"][:, i],
                tau[:, i],
            )
            .toarray()
            .squeeze()
        )
        qddot[:, i] = bio_model.compute_qddot(q[:, i], qdot[:, i], qddot_u_i).toarray().squeeze()
        lambdas[:, i] = (
            compute_lambdas_func(
                q[:, i],
                qdot[:, i],
                qddot[:, i],
                tau[:, i],
            )
            .toarray()
            .squeeze()
        )

    return q, qdot, qddot, lambdas


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int = 30,
    final_time: float = 1,
) -> (HolonomicBiorbdModel, OptimalControlProgram):
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    n_shooting: int
        The number of shooting points
    final_time: float
        The time at the final node

    Returns
    -------
    The ocp ready to be solved
    """
    bio_model = HolonomicBiorbdModel(biorbd_model_path)
    # Create a holonomic constraint to create a double pendulum from two single pendulums
    constraint, constraint_jacobian, constraint_double_derivative = HolonomicConstraintFcn.superimpose_markers(
        bio_model, "marker_1", "marker_3", index=slice(1, 3), local_frame_index=0
    )
    bio_model.add_holonomic_constraint(
        constraint=constraint,
        constraint_jacobian=constraint_jacobian,
        constraint_double_derivative=constraint_double_derivative,
    )
    # The rotations (joint 0 and 3) are independent. The translations (joint 1 and 2) are constrained by the holonomic
    # constraint
    bio_model.set_dependencies(independent_joint_index=[0, 3], dependent_joint_index=[1, 2])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=0.5, max_bound=0.6)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.HOLONOMIC_TORQUE_DRIVEN, expand=False)

    # Path Constraints
    constraints = ConstraintList()

    # Boundaries
    mapping = BiMappingList()
    # The rotations (joint 0 and 3) are independent. The translations (joint 1 and 2) are constrained by the holonomic
    # constraint, so we need to map the states and controls to only compute the dynamics of the independent joints
    # The dynamics of the dependent joints will be computed from the holonomic constraint
    mapping.add("q", to_second=[0, None, None, 1], to_first=[0, 3])
    mapping.add("qdot", to_second=[0, None, None, 1], to_first=[0, 3])
    x_bounds = BoundsList()
    # q_u and qdot_u are the states of the independent joints
    x_bounds["q_u"] = bio_model.bounds_from_ranges("q", mapping=mapping)
    x_bounds["qdot_u"] = bio_model.bounds_from_ranges("qdot", mapping=mapping)

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q_u", [1.54, 1.54])
    x_init.add("qdot_u", [0, 0])
    x_bounds["q_u"][:, 0] = [1.54, 1.54]
    x_bounds["qdot_u"][:, 0] = [0, 0]
    x_bounds["q_u"][0, -1] = -1.54
    x_bounds["q_u"][1, -1] = 0

    # Define control path constraint
    variable_bimapping = BiMappingList()
    tau_min, tau_max, tau_init = -100, 100, 0
    # Only the rotations are controlled
    variable_bimapping.add("tau", to_second=[0, None, None, 1], to_first=[0, 3])
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * 2, max_bound=[tau_max] * 2)
    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * 2)

    # ------------- #

    return (
        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_init=x_init,
            u_init=u_init,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            assume_phase_dynamics=True,
            variable_mappings=variable_bimapping,
            constraints=constraints,
        ),
        bio_model,
    )


def main():
    """
    Runs the optimization and animates it
    """

    model_path = "models/two_pendulums.bioMod"
    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT())

    # --- Show results --- #
    q, qdot, qddot, lambdas = compute_all_states(sol, bio_model)

    import bioviz

    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()

    plt.title("Lagrange multipliers of the holonomic constraint")
    plt.plot(sol.time, lambdas[0, :], label="y")
    plt.plot(sol.time, lambdas[1, :], label="z")
    plt.xlabel("Time (s)")
    plt.ylabel("Lagrange multipliers (N)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
