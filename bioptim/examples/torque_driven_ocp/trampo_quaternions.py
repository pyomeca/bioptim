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
    biorbd_model_path: str, n_shooting: int, final_time: float, ode_solver: OdeSolverBase = OdeSolver.RK4(), assume_phase_dynamics: bool = True,
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
    ode_solver: OdeSolver
        The ode solver to use
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, marker_index=1, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=100)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Define control path constraint
    n_tau = bio_model.nb_tau  # bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)

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
    x_init.add(x, interpolation=InterpolationType.LINEAR)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model.bounds_from_ranges(["q", "qdot"]))
    x_bounds[0].min[: bio_model.nb_q, 0] = x[: bio_model.nb_q, 0]
    x_bounds[0].max[: bio_model.nb_q, 0] = x[: bio_model.nb_q, 0]

    u_init = InitialGuessList()
    u_init.add([tau_init] * n_tau)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        ode_solver=ode_solver,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    Prepares and solves an ocp that has quaternion in it. Animates the results
    """

    ocp = prepare_ocp("models/TruncAnd2Arm_Quaternion.bioMod", n_shooting=5, final_time=0.25)
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # Print the last solution
    sol.animate(n_frames=-1)


if __name__ == "__main__":
    main()
