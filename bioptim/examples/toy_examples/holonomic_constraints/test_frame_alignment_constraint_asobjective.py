#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example: two cubes falling under gravity, kept parallel by a holonomic
constraint on their orientations (the “align_frames” constraint).

"""

import os

import numpy as np
from casadi import DM, MX, vertcat

from bioptim import (
    BiMappingList,
    BoundsList,
    ConstraintList,
    DynamicsOptions,
    DynamicsOptionsList,
    InterpolationType,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    SolutionMerge,
    Solver,
    TorqueBiorbdModel,
    Node,
)
from bioptim.examples.utils import ExampleUtils


def custom_obj_align_frames(controller, frame_1_idx=1, frame_2_idx=2, local_frame_idx=0):
    q_sym = controller.states["q"].cx
    parameters = controller.parameters.cx

    # Homogeneous transformation matrices of the two frames
    # Global homogeneous matrices (4×4) of the two frames
    T1_glob = controller.model.homogeneous_matrices_in_global(segment_index=frame_1_idx)(
        q_sym, parameters
    )  # shape (4,4)
    T2_glob = controller.model.homogeneous_matrices_in_global(segment_index=frame_2_idx)(
        q_sym, parameters
    )  # shape (4,4)

    T_loc = controller.model.homogeneous_matrices_in_global(segment_index=local_frame_idx, inverse=True)(
        q_sym, parameters
    )
    T1_glob = T_loc @ T1_glob
    T2_glob = T_loc @ T2_glob

    # Extract only the 3×3 rotation part (the upper‑left block)
    R1 = T1_glob[:3, :3]  # shape (3,3)
    R2 = T2_glob[:3, :3]  # shape (3,3)

    # Relative rotation: R_rel = R1ᵀ·R2    (frame‑1 → frame‑2)
    R_rel = R1.T @ R2  # 3×3 matrix

    S = (R_rel - R_rel.T) / 2.0
    constraint = vertcat(S[1, 0], S[2, 0], S[2, 1])

    return constraint


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int = 100,
    final_time: float = 1.0,
    expand_dynamics: bool = False,
    ode_solver=OdeSolver.RK4(),
):

    bio_model = TorqueBiorbdModel(biorbd_model_path)

    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1e-2)
    # objectives.add(
    #     custom_obj_align_frames,
    #     custom_type=ObjectiveFcn.Lagrange,
    #     quadratic=True,
    #     weight=100,
    # )
    objectives.add(
        custom_obj_align_frames,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.END,
        quadratic=True,
        weight=100,
    )

    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=ode_solver, expand_dynamics=expand_dynamics))

    x_bounds = BoundsList()
    x_bounds.add(
        "q",
        min_bound=np.array([[-100] * 8, [-100] * 8, [-100] * 8]).T,
        max_bound=np.array([[100] * 8, [100] * 8, [100] * 8]).T,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    x_bounds.add(
        "qdot",
        min_bound=np.array([[-100] * 8, [-100] * 8, [-100] * 8]).T,
        max_bound=np.array([[100] * 8, [100] * 8, [100] * 8]).T,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    # x_bounds["q"][:-3, 0] = [-3, -np.pi / 3, np.pi / 6, 0, 1]
    x_bounds["q"][:, 0] = [1, -np.pi / 3, np.pi / 6, 0, 1, 0, 0, 0]
    x_bounds["qdot"][:, 0] = 0  # no initial velocity
    # x_bounds["qdot"][:-3, 0] = 0  # no initial velocity
    x_bounds["qdot"][1, 0] = 1  # add initial rotation velocity on one axis
    x_bounds["qdot"][2, 0] = 0.5  # add initial rotation velocity on one axis
    x_bounds["qdot"][3, 0] = -0.5

    u_bounds = BoundsList()
    # u_bounds["tau"] = [0] * 8, [0] * 8
    u_bounds["tau"] = [0, 0, 0, 0, 0, -100, -100, -100], [0, 0, 0, 0, 0, 100, 100, 100]

    constraints = ConstraintList()

    ocp = OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objectives,
        constraints=constraints,
        n_threads=24,
    )
    return ocp, bio_model


def main():
    model_folder = os.path.join(ExampleUtils.folder, "models")
    model_path = os.path.join(model_folder, "two_cubes.bioMod")

    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path, n_shooting=100, final_time=1.0)

    ocp.add_plot_penalty()
    solver = Solver.IPOPT()
    solver.set_linear_solver("ma57")
    sol = ocp.solve(solver)

    print(f"Optimization finished in {sol.real_time_to_optimize:.2f} s")

    # --- Show results --- #
    q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]

    viewer = "bioviz"
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

    sol.graphs()


if __name__ == "__main__":
    main()
