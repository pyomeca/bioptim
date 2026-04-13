#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example: two cubes actuated by torques in all 3 directions, kept parallel by a holonomic
constraint on their orientations (the “align_frames” constraint), maintaining a known relative rotation defined by a rotation matrix.

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
    InitialGuessList,
    BiMappingList,
)
from bioptim.examples.utils import ExampleUtils
from custom_dynamics import ModifiedHolonomicTorqueBiorbdModel, constraint_holonomic, constraint_holonomic_end


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
        print(q[:, i])
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
    ode_solver=OdeSolver.RK4(),
):

    # R_desired = np.array(
    #     [
    #         [0, 0, -1],
    #         [0, 1, 0],
    #         [1, 0, 0],
    #     ]
    # )

    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        "align_cubes",
        HolonomicConstraintsFcn.align_frames,
        frame_1_idx=1,  # segment index of the first cube
        frame_2_idx=3,  # segment index of the second cube
        # relative_rotation=R_desired,
    )

    bio_model = ModifiedHolonomicTorqueBiorbdModel(
        biorbd_model_path,
        holonomic_constraints=holonomic_constraints,
        independent_joint_index=[0],
        dependent_joint_index=[1, 2, 3],  # rotation of cube 1
    )

    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)

    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=ode_solver, expand_dynamics=expand_dynamics))

    u_variable_bimapping = BiMappingList()
    u_variable_bimapping.add("q", to_second=[0, None, None, None], to_first=[0])
    u_variable_bimapping.add("qdot", to_second=[0, None, None, None], to_first=[0])

    v_variable_bimapping = BiMappingList()
    v_variable_bimapping.add("q", to_second=[None, 0, 1, 2], to_first=[1, 2, 3])

    x_bounds = BoundsList()
    x_bounds["q_u"] = bio_model.bounds_from_ranges("q", mapping=u_variable_bimapping)
    x_bounds["qdot_u"] = bio_model.bounds_from_ranges("qdot", mapping=u_variable_bimapping)

    x_bounds["q_u"][:, 0] = [0]
    x_bounds["qdot_u"][:, 0] = 0  # no initial velocity

    u_bounds = BoundsList()
    u_bounds["tau"] = [1, 0, 0, 0], [1, 0, 0, 0]

    a_bounds = BoundsList()
    a_bounds.add("q_v", bio_model.bounds_from_ranges("q", mapping=v_variable_bimapping))

    # Initial guess
    x_init = InitialGuessList()
    initial_pos = [0, np.pi / 7, np.pi / 8 + np.pi / 2, -np.pi / 6]
    x_init["q_u"] = [initial_pos[i] for i in [0]]
    a_init = InitialGuessList()
    a_init.add("q_v", [initial_pos[i] for i in [1, 2, 3]])

    constraints = ConstraintList()
    # constraints.add(constraint_holonomic, node=Node.ALL_SHOOTING)
    # constraints.add(constraint_holonomic_end, node=Node.END)

    ocp = OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        a_bounds=a_bounds,
        x_init=x_init,
        a_init=a_init,
        objective_functions=objectives,
        constraints=constraints,
        n_threads=24,
    )
    return ocp, bio_model


def main():
    model_folder = os.path.join(ExampleUtils.folder, "models")
    model_path = os.path.join(model_folder, "two_cubes_lagrange2D_outofplane.bioMod")

    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path, n_shooting=10, final_time=2.0)

    print(bio_model.holonomic_constraints([0, np.pi / 7, np.pi / 8 + np.pi / 2, -np.pi / 6]))
    print(bio_model.holonomic_constraints([0, np.pi / 7, np.pi / 8 - np.pi / 2, -np.pi / 6]))

    solver = Solver.IPOPT()
    sol = ocp.solve(solver)

    print(f"Optimization finished in {sol.real_time_to_optimize:.2f} s")

    # --- Extract Lagrange multipliers ---
    q, _, _, _ = compute_all_states(sol, bio_model)

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
