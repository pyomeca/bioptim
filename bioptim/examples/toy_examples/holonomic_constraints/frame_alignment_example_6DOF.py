#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example: two cubes actuated by torques in all 3 directions, kept parallel by a holonomic
constraint on their orientations (the “align_frames” constraint).

"""

import os

import numpy as np
from casadi import DM

from bioptim import (
    BoundsList,
    ConstraintList,
    DynamicsOptions,
    DynamicsOptionsList,
    HolonomicConstraintsFcn,
    HolonomicConstraintsList,
    HolonomicTorqueBiorbdModel,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    SolutionMerge,
    Solver,
    InterpolationType,
    Node,
    BiMappingList,
    InitialGuessList,
)
from bioptim.examples.utils import ExampleUtils

from custom_dynamics import ModifiedHolonomicTorqueBiorbdModel, constraint_holonomic, constraint_holonomic_end

n_shooting = 30

# Define the three points (each is a 4D vector)
point1 = np.array([0]).T  # -0.1 fails
point2 = np.array([1]).T  # -1.7 fails
point3 = np.array([0]).T
# Generate interpolation points (0 to 2)
t = np.linspace(0, 2, n_shooting)

# Interpolate between point1 and point2 (first half)
interp1 = point1 + t[: n_shooting // 2, np.newaxis] * (point2 - point1)

# Interpolate between point2 and point3 (second half)
interp2 = point2 + t[: n_shooting // 2, np.newaxis] * (point3 - point2)

# Combine the two interpolations
interpolated_points = np.vstack((interp1, interp2))


def compute_all_states(sol, bio_model: HolonomicTorqueBiorbdModel):
    """
    Compute all the states from the solution of the optimal control program

    Parameters
    ----------
    bio_model: HolonomicTorqueBiorbdModel
        The biorbd model
    sol:
        The solution of the optimal control program

    Returns
    -------

    """

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    n = states["q_u"].shape[1]
    n_tau = controls["tau"].shape[1]

    q = np.zeros((bio_model.nb_q, n))
    qdot = np.zeros((bio_model.nb_q, n))
    qddot = np.zeros((bio_model.nb_q, n))
    lambdas = np.zeros((bio_model.nb_dependent_joints, n))
    tau = np.zeros((bio_model.nb_tau, n_tau + 1))

    for independent_joint_index in bio_model.independent_joint_index:
        tau[independent_joint_index, :-1] = controls["tau"][independent_joint_index, :]
    for dependent_joint_index in bio_model.dependent_joint_index:
        tau[dependent_joint_index, :-1] = controls["tau"][dependent_joint_index, :]

    q_v_init = DM.zeros(bio_model.nb_dependent_joints, n)
    for i in range(n):
        q_v_i = bio_model.compute_q_v()(states["q_u"][:, i], q_v_init[:, i]).toarray()
        q[:, i] = bio_model.state_from_partition(states["q_u"][:, i][:, np.newaxis], q_v_i).toarray().squeeze()
        qdot[:, i] = bio_model.compute_qdot()(q[:, i], states["qdot_u"][:, i]).toarray().squeeze()
        qddot_u_i = (
            bio_model.partitioned_forward_dynamics()(
                states["q_u"][:, i], states["qdot_u"][:, i], q_v_init[:, i], tau[:, i]
            )
            .toarray()
            .squeeze()
        )
        qddot[:, i] = bio_model.compute_qddot()(q[:, i], qdot[:, i], qddot_u_i).toarray().squeeze()
        lambdas[:, i] = (
            bio_model.compute_the_lagrangian_multipliers()(
                states["q_u"][:, i][:, np.newaxis], states["qdot_u"][:, i], q_v_init[:, i], tau[:, i]
            )
            .toarray()
            .squeeze()
        )

    return q, qdot, qddot, lambdas


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int = 30,
    final_time: float = 1.0,
    expand_dynamics: bool = False,
    ode_solver=OdeSolver.COLLOCATION(),
):

    # Create a holonomic constraint to create a double pendulum from two single pendulums
    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        "holonomic_constraints",
        HolonomicConstraintsFcn.superimpose_markers,
        marker_1="cube0_1",
        marker_2="cube1_1",
        index=slice(0, 3),
        local_frame_index=1,
    )

    # R_desired = np.array([[1, 0, 0], [0, 0.9848, 0.1736], [0, -0.1736, 0.9848]])
    # R_desired = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # R_desired = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    holonomic_constraints.add(
        "align_cubes",
        HolonomicConstraintsFcn.align_frames3,
        frame_1_idx=1,
        frame_2_idx=9,
        # relative_rotation=R_desired,
    )

    independant_joints = [0, 1, 2, 3, 4, 5]
    computed_joints = [6, 7, 8, 9, 10, 11]

    bio_model = ModifiedHolonomicTorqueBiorbdModel(
        biorbd_model_path,
        holonomic_constraints=holonomic_constraints,
        independent_joint_index=independant_joints,
        dependent_joint_index=computed_joints,
    )
    print([bio_model.model.segments()[i].name().to_string() for i in range(bio_model.nb_segments)])

    # Boundaries
    u_variable_bimapping = BiMappingList()
    u_variable_bimapping.add(
        "q", to_second=[0, 1, 2, 3, 4, 5, None, None, None, None, None, None], to_first=independant_joints
    )
    u_variable_bimapping.add(
        "qdot", to_second=[0, 1, 2, 3, 4, 5, None, None, None, None, None, None], to_first=independant_joints
    )

    v_variable_bimapping = BiMappingList()
    v_variable_bimapping.add(
        "q",
        to_second=[None, None, None, None, None, None, 0, 1, 2, 3, 4, 5],
        to_first=computed_joints,
    )

    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objectives.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q_u", index=[0], target=interpolated_points.T, weight=10000)
    objectives.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q_u", index=[4], target=interpolated_points.T, weight=1000)

    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=ode_solver, expand_dynamics=expand_dynamics))

    # Path bounds
    x_bounds = BoundsList()
    x_bounds["q_u"] = bio_model.bounds_from_ranges("q", mapping=u_variable_bimapping)
    x_bounds["q_u"][:, 0] = 0  # Start and end without any velocity

    x_bounds["qdot_u"] = bio_model.bounds_from_ranges("qdot", mapping=u_variable_bimapping)
    x_bounds["qdot_u"][:, [0, -1]] = 0  # Start and end without any velocity

    # tau_variable_bimapping = BiMappingList()
    # tau_variable_bimapping.add(
    #     "tau", to_second=[0, 1, 2, 3, 4, 5, None, None, None, None, None, None], to_first=independant_joints
    # )

    u_bounds = BoundsList()
    u_bounds["tau"] = [-100, -100, -100, -100, -100, -100, 0, 0, 0, 0, 0, 0], [
        100,
        100,
        100,
        100,
        100,
        100,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    initial_pos = [0, 0, 0, 0, 0, 0, -1.5, 0, -1.5, 0, 0, 0]

    x_init = InitialGuessList()
    x_init["q_u"] = [initial_pos[i] for i in independant_joints]
    a_init = InitialGuessList()
    a_init.add("q_v", [initial_pos[i] for i in computed_joints])

    # Path Constraints
    constraints = ConstraintList()
    constraints.add(constraint_holonomic, node=Node.ALL_SHOOTING)
    constraints.add(constraint_holonomic_end, node=Node.END)

    ocp = OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        a_init=a_init,
        objective_functions=objectives,
        constraints=constraints,
        n_threads=24,
    )
    return ocp, bio_model


def main():
    model_folder = os.path.join(ExampleUtils.folder, "models")
    model_path = os.path.join(model_folder, "two_cubes_lagrange2D_6DOF.bioMod")

    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path, n_shooting=n_shooting, final_time=1.0)

    solver = Solver.IPOPT()
    sol = ocp.solve(solver)

    print(f"Optimization finished in {sol.real_time_to_optimize:.2f} s")

    # --- Extract Lagrange multipliers ---
    # q, _, _, _ = compute_all_states(sol, bio_model)
    stepwise_q_u = sol.stepwise_states(to_merge=SolutionMerge.NODES)["q_u"]
    stepwise_q_v = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)["q_v"]
    q = ocp.nlp[0].model.state_from_partition(stepwise_q_u, stepwise_q_v).toarray()

    viewer = "pyorerun"
    if viewer == "bioviz":
        import bioviz

        viz = bioviz.Viz(model_path)
        viz.show_global_ref_frame = True
        viz.load_movement(q)
        viz.exec()

    if viewer == "pyorerun":
        import pyorerun

        viz = pyorerun.PhaseRerun(t_span=np.concatenate(sol.decision_time()).squeeze())
        viz.add_animated_model(pyorerun.BiorbdModel(model_path), q=q)

        viz.rerun("double_pendulum")

    sol.graphs()


if __name__ == "__main__":
    main()
