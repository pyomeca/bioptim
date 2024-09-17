"""
This example uses a representation of a human body by a trunk-leg segment and two arms which orientation is represented
using quaternions.
The goal of the OCP is to elevate the position of the trunk in a environment without gravity with minimal efforts.
It is designed to show how to use a model that has quaternions in their degrees of freedom.
"""

import platform

import biorbd_casadi as biorbd
import numpy as np
from casadi import MX, Function

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    InterpolationType,
    OdeSolver,
    OdeSolverBase,
    Node,
    Solver,
    PhaseDynamics,
    ConstraintList,
    ConstraintFcn,
)


def eul2quat(eul: np.ndarray) -> np.ndarray:
    """
    Converts Euler angles to quaternion. It assumes a sequence angle of XYZ

    Parameters
    ----------
    eul: np.ndarray
        The 3 angles of sequence XYZ

    Returns
    -------
    The quaternion associated to the Euler angles in the format [W, X, Y, Z]
    """
    eul_sym = MX.sym("eul", 3)
    Quat = Function("Quaternion_fromEulerAngles", [eul_sym], [biorbd.Quaternion.fromXYZAngles(eul_sym).to_mx()])(eul)
    return Quat


def quat2eul(quat: np.ndarray) -> np.ndarray:
    """
    Converts quaternion to Euler angles. It assumes a sequence angle of XYZ

    Parameters
    ----------
    quat: np.ndarray
        The quaternion in the format [W, X, Y, Z]

    Returns
    -------
    The Euler angles associated to the quaternion in the format [X, Y, Z]
    """
    quat_sym = MX.sym("quat", 4)
    quat_biorbd = biorbd.Quaternion(quat_sym[3], quat_sym[0], quat_sym[1], quat_sym[2])
    eul_mx = biorbd.Rotation.toEulerAngles(biorbd.Quaternion.toMatrix(quat_biorbd), "xyz").to_mx()
    eul = Function("EulerAngles_fromQuaternion", [quat_sym], [eul_mx])(quat)
    return eul


def euler_dot2omega(eul: np.ndarray, eul_dot: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """
    Converts Euler angle rates to body velocity.

    Parameters
    ----------
    eul: np.ndarray
        The 3 angles of sequence XYZ
    eul_dot: np.ndarray
        The 3 angle rates of sequence XYZ
    quat: np.ndarray
        The associated quaternion

    Returns
    -------
    The angular velocity associated to the Euler angles in the format [X, Y, Z]
    """

    eul_sym = MX.sym("eul", 3)
    eul_dot_sym = MX.sym("eul_dot", 3)
    quat_sym = MX.sym("quat", 4)
    quat_biorbd = biorbd.Quaternion(quat_sym[3], quat_sym[0], quat_sym[1], quat_sym[2])
    omega_mx = biorbd.Quaternion.eulerDotToOmega(quat_biorbd, eul_sym, eul_dot_sym, "xyz").to_mx()
    omega = Function("omega", [quat_sym, eul_sym, eul_dot_sym], [omega_mx])(quat, eul, eul_dot)
    return omega


def joint_angles_rate2body_velcities(q: np.ndarray, eul_dot: np.ndarray) -> np.ndarray:
    """
    Converts joint angle rate to body velocity because of quaternions.

    Parameters
    ----------
    q: np.ndarray
        The generalized coordinates
    eul_dot: np.ndarray
        The desired Euler joint angle rate

    Returns
    -------
    The body velocities
    """
    right_arm_omega = np.array(
        euler_dot2omega(eul=quat2eul(q[[6, 7, 8, 12]]), eul_dot=eul_dot[6:9], quat=q[[6, 7, 8, 12]])
    ).reshape(
        -1,
    )
    left_arm_omega = np.array(
        euler_dot2omega(eul=quat2eul(q[[9, 10, 11, 13]]), eul_dot=eul_dot[9:12], quat=q[[9, 10, 11, 13]])
    ).reshape(-1)
    qdot = np.hstack((right_arm_omega, left_arm_omega))
    return qdot


def define_x_init(bio_model) -> np.ndarray:
    """
    Defines the initial guess for the states.
    The intial guess for the quaternion of the arms are based on the positions of the arms in Euler angles.
    """
    x = np.vstack((np.zeros((bio_model.nb_q, 2)), np.ones((bio_model.nb_qdot, 2))))
    right_arm_init = np.zeros((3, 2))
    right_arm_init[1, 0] = -np.pi + 0.01
    right_arm_init[1, 1] = 0
    left_arm_init = np.zeros((3, 2))
    left_arm_init[1, 0] = np.pi - 0.01
    left_arm_init[1, 1] = 0
    for i in range(2):
        right_arm_quaterion = eul2quat(right_arm_init[:, i])
        left_arm_quaterion = eul2quat(left_arm_init[:, i])
        x[6:9, i] = np.reshape(right_arm_quaterion[1:], 3)
        x[12, i] = right_arm_quaterion[0]
        x[9:12, i] = np.reshape(left_arm_quaterion[1:], 3)
        x[13, i] = left_arm_quaterion[0]
    return x


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod file
    n_shooting: int
        The number of shooting points
    final_time: float
        The time at the final node
    ode_solver: OdeSolverBase
        The ode solver to use
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau_joints", node=Node.ALL_SHOOTING, weight=100
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1)

    # Add constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="Target_START", second_marker="Neck"
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="Target_END",
        second_marker="Neck",
        min_bound=0,
        max_bound=np.inf,
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.TORQUE_DRIVEN_FREE_FLOATING_BASE, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics
    )

    # Define control path constraint
    n_root = bio_model.nb_root
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    n_tau = bio_model.nb_tau - n_root
    tau_min, tau_max = -100, 100
    u_bounds = BoundsList()
    u_bounds["tau_joints"] = [tau_min] * n_tau, [tau_max] * n_tau

    # Initial guesses
    x_init = InitialGuessList()
    x = define_x_init(bio_model)
    x_init.add("q_roots", x[:n_root, :], interpolation=InterpolationType.LINEAR)
    x_init.add("q_joints", x[n_root:n_q, :], interpolation=InterpolationType.LINEAR)
    x_init.add("qdot_roots", x[n_q : n_q + n_root, :], interpolation=InterpolationType.LINEAR)
    x_init.add("qdot_joints", x[n_q + n_root :, :], interpolation=InterpolationType.LINEAR)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q_roots"] = bio_model.bounds_from_ranges("q_roots")
    x_bounds["q_joints"] = bio_model.bounds_from_ranges("q_joints")
    x_bounds["qdot_roots"] = bio_model.bounds_from_ranges("qdot_roots")
    x_bounds["qdot_joints"] = bio_model.bounds_from_ranges("qdot_joints")
    x_bounds["q_roots"][:, 0] = 0
    x_bounds["qdot_roots"][:, 0] = 0
    x_bounds["q_joints"][:, 0] = x_init["q_joints"].init[:, 0]
    omega_arms = joint_angles_rate2body_velcities(
        np.hstack((np.zeros((n_root,)), x_init["q_joints"].init[:, 0])), np.zeros((n_qdot,))
    )
    x_bounds["qdot_joints"].min[:, 0] = omega_arms - 0.1
    x_bounds["qdot_joints"].max[:, 0] = omega_arms + 0.1

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
    )


def main():
    """
    Prepares and solves an ocp that has quaternion in it. Animates the results
    """

    n_shooting = 6
    ocp = prepare_ocp("models/trunk_and_2arm_quaternion.bioMod", n_shooting=n_shooting, final_time=0.25)
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    # sol.graphs()
    # If you get an error message in animate with quaternions, it is due to the interpolation of quaternions in bioviz.
    # To avoid problems, specify the number of frames to be the same as the number of shooting points
    sol.animate(n_frames=n_shooting + 1, show_gravity_vector=False)


if __name__ == "__main__":
    main()
