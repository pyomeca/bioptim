"""
TODO: Create a more meaningful example (make sure to translate all the variables [they should correspond to the model])
This example uses a representation of a human body by a trunk_leg segment and two arms and has the objective to...
It is designed to show how to use a model that has quaternions in their degrees of freedom.
"""

import platform

import numpy as np
import biorbd_casadi as biorbd
from casadi import MX, Function
from bioptim import (
    TorqueBiorbdModel,
    OptimalControlProgram,
    DynamicsOptionsList,
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
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = TorqueBiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, marker_index=1, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=100)

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(ode_solver=ode_solver, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics
    )

    # Define control path constraint
    n_tau = bio_model.nb_tau  # bio_model.nb_tau
    tau_min, tau_max = -100, 100
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau

    # Initial guesses
    # TODO put this in a function defined before and explain what it does, and what are the variables
    x = np.vstack((np.zeros((bio_model.nb_q, 2)), np.ones((bio_model.nb_qdot, 2))))
    Arm_init_D = np.zeros((3, 2))
    Arm_init_D[1, 0] = 0
    Arm_init_D[1, 1] = -np.pi + 0.01
    Arm_init_G = np.zeros((3, 2))
    Arm_init_G[1, 0] = 0
    Arm_init_G[1, 1] = np.pi - 0.01
    for i in range(2):
        Arm_Quat_D = eul2quat(Arm_init_D[:, i])
        Arm_Quat_G = eul2quat(Arm_init_G[:, i])
        x[6:9, i] = np.reshape(Arm_Quat_D[1:], 3)
        x[12, i] = Arm_Quat_D[0]
        x[9:12, i] = np.reshape(Arm_Quat_G[1:], 3)
        x[13, i] = Arm_Quat_G[0]
    x_init = InitialGuessList()
    x_init.add("q", x[: bio_model.nb_q, :], interpolation=InterpolationType.LINEAR)
    x_init.add("qdot", x[bio_model.nb_q :, :], interpolation=InterpolationType.LINEAR)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"].min[: bio_model.nb_q, 0] = x[: bio_model.nb_q, 0]
    x_bounds["q"].max[: bio_model.nb_q, 0] = x[: bio_model.nb_q, 0]
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
    )


def main():
    """
    Prepares and solves an ocp that has quaternion in it. Animates the results
    """

    ocp = prepare_ocp("models/trunk_and_2arm_quaternion.bioMod", n_shooting=25, final_time=0.25)
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"), expand_during_shake_tree=False)

    # Print the last solution
    sol.animate(n_frames=-1)


if __name__ == "__main__":
    main()
