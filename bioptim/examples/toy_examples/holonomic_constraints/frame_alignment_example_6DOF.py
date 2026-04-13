#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example: two cubes actuated by torques in all 3 directions, kept parallel by a holonomic
constraint on their orientations (the “align_frames_generalized” constraint).

"""

import os

import numpy as np

from bioptim import (
    BoundsList,
    ConstraintList,
    DynamicsOptions,
    DynamicsOptionsList,
    HolonomicConstraintsFcn,
    HolonomicConstraintsList,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    SolutionMerge,
    Solver,
    Node,
    BiMappingList,
    InitialGuessList,
)
from bioptim.examples.utils import ExampleUtils

from .custom_dynamics import ModifiedHolonomicTorqueBiorbdModel, constraint_holonomic, constraint_holonomic_end

n_shooting = 30

# Define the three points (each is a 4D vector)
point1 = np.array([0]).T
point2 = np.array([1]).T
point3 = np.array([0]).T
# Generate interpolation points (0 to 2)
t = np.linspace(0, 2, n_shooting)

# Interpolate between point1 and point2 (first half)
interp1 = point1 + t[: n_shooting // 2, np.newaxis] * (point2 - point1)

# Interpolate between point2 and point3 (second half)
interp2 = point2 + t[: n_shooting // 2, np.newaxis] * (point3 - point2)

# Combine the two interpolations
interpolated_points = np.vstack((interp1, interp2))


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

    holonomic_constraints.add(
        "align_cubes",
        HolonomicConstraintsFcn.align_frames_generalized,
        frame_1_idx=1,
        frame_2_idx=9,
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
    objectives.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q_u", index=[0], target=interpolated_points.T, weight=1e4)
    objectives.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q_u", index=[4], target=interpolated_points.T, weight=1e3)

    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=ode_solver, expand_dynamics=expand_dynamics))

    # Path bounds
    x_bounds = BoundsList()
    x_bounds["q_u"] = bio_model.bounds_from_ranges("q", mapping=u_variable_bimapping)
    x_bounds["q_u"][:, 0] = 0  # Start pos

    x_bounds["qdot_u"] = bio_model.bounds_from_ranges("qdot", mapping=u_variable_bimapping)

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
    )
    return ocp, bio_model


def main():
    model_folder = os.path.join(ExampleUtils.folder, "models")
    model_path = os.path.join(model_folder, "two_cubes_lagrange2D_6DOF.bioMod")

    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path, n_shooting=n_shooting, final_time=1.0)

    solver = Solver.IPOPT()
    sol = ocp.solve(solver)

    print(f"Optimization finished in {sol.real_time_to_optimize:.2f} s")

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    print(states["q_u"])

    # --- Extract Lagrange multipliers ---
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
