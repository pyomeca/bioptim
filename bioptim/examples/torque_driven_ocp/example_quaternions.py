"""
This example uses a representation of a human body by a trunk-leg segment and two arms which orientation is represented
using quaternions.
The goal of the OCP is to elevate the position of the trunk in a environment without gravity with minimal efforts.
It is designed to show how to use a model that has quaternions in their degrees of freedom.
"""

import platform

import numpy as np
import biorbd_casadi as biorbd
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

def define_x_init(bio_model) -> np.ndarray:
    """
    Defines the initial guess for the states.
    The intial guess for the quaternion of the arms are based on the positions of the arms in Euler angles.
    """
    x = np.vstack((np.zeros((bio_model.nb_q, 2)), np.ones((bio_model.nb_qdot, 2))))
    right_arm_init = np.zeros((3, 2))
    right_arm_init[1, 0] = 0
    right_arm_init[1, 1] = -np.pi + 0.01
    left_arm_init = np.zeros((3, 2))
    left_arm_init[1, 0] = 0
    left_arm_init[1, 1] = np.pi - 0.01
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
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau_joints", node=Node.ALL_SHOOTING, weight=100)

    # Add constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN_FREE_FLOATING_BASE,
                 expand_dynamics=expand_dynamics,
                 phase_dynamics=phase_dynamics)

    # Define control path constraint
    n_root = bio_model.nb_root
    n_q = bio_model.nb_q
    n_tau = bio_model.nb_tau - n_root
    tau_min, tau_max = -100, 100
    u_bounds = BoundsList()
    u_bounds["tau_joints"] = [tau_min] * n_tau, [tau_max] * n_tau

    # Initial guesses
    x_init = InitialGuessList()
    x = define_x_init(bio_model)
    x_init.add("q_roots", x[: n_root, :], interpolation=InterpolationType.LINEAR)
    x_init.add("q_joints", x[n_root: n_q, :], interpolation=InterpolationType.LINEAR)
    x_init.add("qdot_roots", x[n_q: n_q + n_root, :], interpolation=InterpolationType.LINEAR)
    x_init.add("qdot_joints", x[n_q + n_root:, :], interpolation=InterpolationType.LINEAR)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q_roots"] = bio_model.bounds_from_ranges("q_roots")
    x_bounds["q_joints"] = bio_model.bounds_from_ranges("q_joints")
    x_bounds["qdot_roots"] = bio_model.bounds_from_ranges("qdot_roots")
    x_bounds["qdot_joints"] = bio_model.bounds_from_ranges("qdot_joints")

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
    )


def main():
    """
    Prepares and solves an ocp that has quaternion in it. Animates the results
    """

    ocp = prepare_ocp("models/trunk_and_2arm_quaternion.bioMod", n_shooting=25, final_time=0.25)
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # Print the last solution
    sol.animate(n_frames=-1)


if __name__ == "__main__":
    main()
