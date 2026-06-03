#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example: two cubes actuated by torques in all 3 directions, kept parallel by a holonomic
constraint on their orientations (the “align_frames” constraint), maintaining a known relative rotation defined by a rotation matrix.

"""

import os

import numpy as np
from casadi import MX, sqrt, trace, vertcat

from bioptim import (
    BoundsList,
    CostType,
    DynamicsOptions,
    DynamicsOptionsList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    PenaltyController,
    SolutionMerge,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils


def custom_objective_align_frames(controller: PenaltyController, frame_1_idx, frame_2_idx, relative_rotation=None):
    # Ensure torque_ref is a CasADi MX
    relative_rotation_mx = MX(relative_rotation)

    q_sym = controller.states["q"].cx
    parameters = controller.parameters.cx

    # If no local frame is specified, use the global frame
    R1 = controller.model.homogeneous_matrices_in_global(segment_index=frame_1_idx)(q_sym, parameters)[:3, :3]
    R2 = controller.model.homogeneous_matrices_in_global(segment_index=frame_2_idx)(q_sym, parameters)[:3, :3]

    # Relative rotation: R_rel = R1ᵀ·R2    (frame‑1 → frame‑2)
    R_rel = relative_rotation_mx.T @ R1.T @ R2  # still a symbolic 3×3 matrix

    # Minimal set of scalar constraints (3 equations)
    # The skew‑symmetric part of a proper rotation is zero when the angle is zero:
    #   S = (R_rel - R_relᵀ) / 2  →  S = 0   ⇔   ω = 0, θ = 0
    # We vectorise the three independent components of S:
    #    S_21, S_31, S_32
    # (any consistent ordering works, we keep the same order used in the
    #  analytical derivation of the constraint in the OP.)
    cos_theta = (trace(R_rel) - 1) / 2
    theta = sqrt(2 * (1 - cos_theta) + 1e-12)  # using the first-order expansion of arccos
    theta_over_sintheta = 1 + theta**2 / 6 + 7 * theta**4 / 360 + 31 * theta**6 / 15120  # using the Taylor expansion
    S = theta_over_sintheta * (R_rel - R_rel.T) / 2.0  # still 3×3, skew‑symmetric

    constraint = vertcat(S[2, 1], -S[2, 0], S[1, 0])  # r32 - r23  # r13 - r31  # r21 - r12

    return constraint


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int = 30,
    final_time: float = 1.0,
    expand_dynamics: bool = False,
    ode_solver=OdeSolver.RK4(),
):

    R_desired = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    bio_model = TorqueBiorbdModel(biorbd_model_path)

    objectives = ObjectiveList()
    # objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objectives.add(
        custom_objective_align_frames,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        quadratic=True,
        frame_1_idx=1,
        frame_2_idx=2,
        relative_rotation=R_desired,
    )

    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=ode_solver, expand_dynamics=expand_dynamics))

    x_bounds = BoundsList()
    x_bounds.add(
        "q",
        min_bound=np.array([[-100] * 4, [-100] * 4, [-100] * 4]).T,
        max_bound=np.array([[100] * 4, [100] * 4, [100] * 4]).T,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    x_bounds.add(
        "qdot",
        min_bound=np.array([[-100] * 4, [-100] * 4, [-100] * 4]).T,
        max_bound=np.array([[100] * 4, [100] * 4, [100] * 4]).T,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    x_bounds["q"][0, 0] = 0
    x_bounds["qdot"][:, 0] = 0  # no initial velocity

    # Initial guess
    # x_init = InitialGuessList()
    # initial_pos = [0, np.pi / 7, np.pi / 8 + np.pi / 2, -np.pi / 6]
    # x_init["q_u"] = [initial_pos[i] for i in [0]]
    # a_init = InitialGuessList()
    # a_init.add("q_v", [initial_pos[i] for i in [1, 2, 3]])

    u_bounds = BoundsList()
    u_bounds["tau"] = [1, -10, -10, -10], [1, 10, 10, 10]

    ocp = OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objectives,
        n_threads=24,
    )
    return ocp, bio_model


def main():
    model_folder = os.path.join(ExampleUtils.folder, "models")
    model_path = os.path.join(model_folder, "two_cubes_lagrange2D_outofplane.bioMod")

    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path, n_shooting=10, final_time=2.0)
    ocp.add_plot_penalty(CostType.ALL)

    solver = Solver.IPOPT()
    sol = ocp.solve(solver)

    print(f"Optimization finished in {sol.real_time_to_optimize:.2f} s")

    q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]

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
