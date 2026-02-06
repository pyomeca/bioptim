#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example: two cubes falling under gravity, kept parallel by a holonomic
constraint on their orientations (the “align_frames” constraint).

"""

import os

import numpy as np
from casadi import DM

from bioptim import (
    BiMappingList,
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
    InitialGuessList,
)
from bioptim.examples.utils import ExampleUtils


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

    for i, independent_joint_index in enumerate(bio_model.independent_joint_index):
        tau[independent_joint_index, :-1] = controls["tau"][i, :]
    for i, dependent_joint_index in enumerate(bio_model.dependent_joint_index):
        tau[dependent_joint_index, :-1] = controls["tau"][i, :]

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


import numpy as np
from casadi import DM


# dummy model that returns the identity rotation for every segment
class DummyModel:
    nb_q = 6
    nb_qdot = 6
    parameters = DM([])

    def homogeneous_matrices_in_global(self, segment_index):
        # returns a function that ignores q and parameters and always gives I4
        return lambda q, p: DM.eye(4)


model = DummyModel()

# Desired relative rotation = 180° about the x‑axis
R_des = DM([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

c_f, c_j, c_dd = HolonomicConstraintsFcn.align_frames3(
    model,
    frame_1_idx=1,
    frame_2_idx=2,
    relative_rotation=R_des,
)

# evaluate the constraint at the (only) configuration
print(c_f(q=DM.zeros(6)))  # → should be [π, 0, 0]ᵀ (non‑zero, i.e. error)
print(c_j(q=DM.zeros(6)))  # Jacobian (3×6 matrix) – just to check it builds


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int = 30,
    final_time: float = 1.0,
    expand_dynamics: bool = False,
    ode_solver=OdeSolver.RK4(),
):

    # Define the desired relative rotation (e.g., a rotation of pi/4 around Z)
    desired_angle = np.pi / 4  # Example: pi/4 radians around Z
    # R_desired = np.array(
    #     [
    #         [np.cos(desired_angle), -np.sin(desired_angle), 0],
    #         [np.sin(desired_angle), np.cos(desired_angle), 0],
    #         [0, 0, 1],
    #     ]
    # )
    R_desired = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]
    )

    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        "align_cubes",
        HolonomicConstraintsFcn.align_frames3,
        frame_1_idx=1,  # segment index of the first cube
        frame_2_idx=2,  # segment index of the second cube
        relative_rotation=R_desired,
        # local_frame_idx=0,
    )

    ## @todo Also test example with xyz constraint on top

    bio_model = HolonomicTorqueBiorbdModel(
        biorbd_model_path,
        holonomic_constraints=holonomic_constraints,
        independent_joint_index=[0, 1, 2, 3, 4],
        dependent_joint_index=[5, 6, 7],  # rotations of cube 1
    )

    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    # objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qdot", weight=1e-2)

    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=ode_solver, expand_dynamics=expand_dynamics))

    # Mapping
    # variable_bimapping = BiMappingList()
    # variable_bimapping.add("q", to_second=[0, 1, 2, 3, 4, None, None, None], to_first=[0, 1, 2, 3, 4])
    # variable_bimapping.add("qdot", to_second=[0, 1, 2, 3, 4, None, None, None], to_first=[0, 1, 2, 3, 4])

    x_bounds = BoundsList()
    x_bounds.add(
        "q_u",
        min_bound=np.array([[-100] * 5, [-100] * 5, [-100] * 5]).T,
        max_bound=np.array([[100] * 5, [100] * 5, [100] * 5]).T,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    x_bounds.add(
        "qdot_u",
        min_bound=np.array([[-100] * 5, [-100] * 5, [-100] * 5]).T,
        max_bound=np.array([[100] * 5, [100] * 5, [100] * 5]).T,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    # x_bounds["q"][:-3, 0] = [-3, -np.pi / 3, np.pi / 6, 0, 1]
    x_bounds["q_u"][:, 0] = [0, -np.pi / 3, np.pi / 6, 0, 1]
    x_bounds["qdot_u"][:, 0] = 0  # no initial velocity
    x_bounds["qdot_u"][1, 0] = 1  # add initial rotation velocity on one axis
    x_bounds["qdot_u"][2, 0] = 0.5  # add initial rotation velocity on one axis
    x_bounds["qdot_u"][3, 0] = -0.5

    # Initial guess
    x_init = InitialGuessList()
    x_init["q_u"] = [0, 1.0, 0, 0, 0]

    # variable_bimapping.add("tau", to_second=[0, 1, 2, 3, 4, None, None, None], to_first=[0, 1, 2, 3, 4])

    u_bounds = BoundsList()
    u_bounds["tau"] = [0] * 8, [0] * 8
    # u_bounds["tau"] = [0, 0, 0, 0, 0, -100, -100, -100], [0, 0, 0, 0, 0, 100, 100, 100]

    constraints = ConstraintList()

    ocp = OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objectives,
        # variable_mappings=variable_bimapping,
        constraints=constraints,
        n_threads=24,
    )
    return ocp, bio_model


def main():
    model_folder = os.path.join(ExampleUtils.folder, "models")
    model_path = os.path.join(model_folder, "two_cubes.bioMod")

    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path, n_shooting=100, final_time=1.0)

    q0 = np.zeros((bio_model.nb_q, 1))
    q0 = np.array([0, 1.0, 0, 0, 0, 0, 0, 0])
    print("c(q0) = ", bio_model.holonomic_constraints(q0))
    R1 = bio_model.homogeneous_matrices_in_global(segment_index=1)(q0, DM([]))[:3, :3]
    print("R1·R1ᵀ =", (R1 @ R1.T).full())

    solver = Solver.IPOPT(show_online_optim=True)
    solver.set_linear_solver("ma57")
    sol = ocp.solve(solver)

    print(f"Optimization finished in {sol.real_time_to_optimize:.2f} s")

    # --- Show results --- #
    q, _, _, _ = compute_all_states(sol, bio_model)

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
