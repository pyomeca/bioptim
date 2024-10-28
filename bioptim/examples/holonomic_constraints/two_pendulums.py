"""
This example presents how to implement a holonomic constraint in bioptim.
The simulation is two single pendulum that are forced to be coherent with a holonomic constraint. It is then a double
pendulum simulation.
"""

import platform

import matplotlib.pyplot as plt
import numpy as np
from casadi import DM

from bioptim import (
    BiMappingList,
    BoundsList,
    ConstraintList,
    DynamicsFcn,
    DynamicsList,
    HolonomicBiorbdModel,
    HolonomicConstraintsFcn,
    HolonomicConstraintsList,
    InitialGuessList,
    ObjectiveFcn,
    ObjectiveList,
    OptimalControlProgram,
    Solver,
    SolutionMerge,
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

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    n = states["q_u"].shape[1]

    q = np.zeros((bio_model.nb_q, n))
    qdot = np.zeros((bio_model.nb_q, n))
    qddot = np.zeros((bio_model.nb_q, n))
    lambdas = np.zeros((bio_model.nb_dependent_joints, n))
    tau = np.zeros((bio_model.nb_tau, n))

    for i, independent_joint_index in enumerate(bio_model.independent_joint_index):
        tau[independent_joint_index, :-1] = controls["tau"][i, :]
    for i, dependent_joint_index in enumerate(bio_model.dependent_joint_index):
        tau[dependent_joint_index, :-1] = controls["tau"][i, :]

    q_v_init = DM.zeros(bio_model.nb_dependent_joints)
    for i in range(n):
        q_v_i = bio_model.compute_q_v()(states["q_u"][:, i], q_v_init).toarray()
        q[:, i] = bio_model.state_from_partition(states["q_u"][:, i][:, np.newaxis], q_v_i).toarray().squeeze()
        qdot[:, i] = bio_model.compute_qdot()(q[:, i], states["qdot_u"][:, i]).toarray().squeeze()
        qddot_u_i = (
            bio_model.partitioned_forward_dynamics()(states["q_u"][:, i], states["qdot_u"][:, i], q_v_init, tau[:, i])
            .toarray()
            .squeeze()
        )
        qddot[:, i] = bio_model.compute_qddot()(q[:, i], qdot[:, i], qddot_u_i).toarray().squeeze()
        lambdas[:, i] = bio_model.compute_the_lagrangian_multipliers()(states["q_u"][:, i][:, np.newaxis], states["qdot_u"][:, i], q_v_init[:, i], tau[:, i]).toarray().squeeze()

    return q, qdot, qddot, lambdas


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int = 30,
    final_time: float = 1,
    expand_dynamics: bool = False,
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
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The ocp ready to be solved
    """
    bio_model = HolonomicBiorbdModel(biorbd_model_path)
    # Create a holonomic constraint to create a double pendulum from two single pendulums
    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        "holonomic_constraints",
        HolonomicConstraintsFcn.superimpose_markers,
        biorbd_model=bio_model,
        marker_1="marker_1",
        marker_2="marker_3",
        index=slice(1, 3),
        local_frame_index=0,
    )
    # The rotations (joint 0 and 3) are independent. The translations (joint 1 and 2) are constrained by the holonomic
    # constraint
    bio_model.set_holonomic_configuration(
        constraints_list=holonomic_constraints, independent_joint_index=[0, 3], dependent_joint_index=[1, 2]
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=0.5, max_bound=0.6)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.HOLONOMIC_TORQUE_DRIVEN, expand_dynamics=expand_dynamics)

    # Path Constraints
    constraints = ConstraintList()

    # Boundaries
    variable_bimapping = BiMappingList()
    # The rotations (joint 0 and 3) are independent. The translations (joint 1 and 2) are constrained by the holonomic
    # constraint, so we need to map the states and controls to only compute the dynamics of the independent joints
    # The dynamics of the dependent joints will be computed from the holonomic constraint
    variable_bimapping.add("q", to_second=[0, None, None, 1], to_first=[0, 3])
    variable_bimapping.add("qdot", to_second=[0, None, None, 1], to_first=[0, 3])
    x_bounds = BoundsList()
    # q_u and qdot_u are the states of the independent joints
    x_bounds["q_u"] = bio_model.bounds_from_ranges("q", mapping=variable_bimapping)
    x_bounds["qdot_u"] = bio_model.bounds_from_ranges("qdot", mapping=variable_bimapping)

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q_u", [1.54, 1.54])
    x_init.add("qdot_u", [0, 0])
    x_bounds["q_u"][:, 0] = [1.54, 1.54]
    x_bounds["qdot_u"][:, 0] = [0, 0]
    x_bounds["q_u"][0, -1] = -1.54
    x_bounds["q_u"][1, -1] = 0

    # Define control path constraint
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
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objective_functions,
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
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    sol.graphs()
    q, qdot, qddot, lambdas = compute_all_states(sol, bio_model)

    viewer = "pyorerun"
    if viewer == "bioviz":
        import bioviz

        viz = bioviz.Viz(model_path)
        viz.load_movement(q)
        viz.exec()

    if viewer == "pyorerun":
        import pyorerun

        viz = pyorerun.PhaseRerun(t_span=np.concatenate(sol.decision_time()).squeeze())
        viz.add_animated_model(pyorerun.BiorbdModel(model_path), q=q)

        viz.rerun("double_pendulum")

    time = sol.decision_time(to_merge=SolutionMerge.NODES)
    plt.title("Lagrange multipliers of the holonomic constraint")
    plt.plot(time, lambdas[0, :], label="y")
    plt.plot(time, lambdas[1, :], label="z")
    plt.xlabel("Time (s)")
    plt.ylabel("Lagrange multipliers (N)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
