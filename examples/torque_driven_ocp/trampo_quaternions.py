"""
TODO: Create a more meaningful example (make sure to translate all the variables [they should correspond to the model])
This example uses a representation of a human body by a trunk_leg segment and two arms and has the objective to...
It is designed to show how to use a model that has quaternions in their degrees of freedom.
"""


import numpy as np
import biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    InterpolationType,
    OdeSolver,
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

    ph = eul[0]
    th = eul[1]
    ps = eul[2]
    cph = np.cos(ph * 0.5)
    sph = np.sin(ph * 0.5)
    cth = np.cos(th * 0.5)
    sth = np.sin(th * 0.5)
    cps = np.cos(ps * 0.5)
    sps = np.sin(ps * 0.5)
    w = -sph * sth * sps + cph * cth * cps
    x = sph * cth * cps + cph * sth * sps
    y = cph * sth * cps - sph * cth * sps
    z = sph * sth * cps + cph * cth * sps
    return np.array([w, x, y, z])


def prepare_ocp(
    biorbd_model_path: str, n_shooting: int, final_time: float, ode_solver: OdeSolver = OdeSolver.RK4()
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

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, index=1, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, name="tau", weight=100)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Define control path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()  # biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)

    # Initial guesses
    # TODO put this in a function defined before and explain what it does, and what are the variables
    x = np.vstack((np.zeros((biorbd_model.nbQ(), 2)), np.ones((biorbd_model.nbQdot(), 2))))
    Arm_init_D = np.zeros((3, 2))
    Arm_init_D[1, 0] = 0
    Arm_init_D[1, 1] = -np.pi + 0.01
    Arm_init_G = np.zeros((3, 2))
    Arm_init_G[1, 0] = 0
    Arm_init_G[1, 1] = np.pi - 0.01
    for i in range(2):
        Arm_Quat_D = eul2quat(Arm_init_D[:, i])
        Arm_Quat_G = eul2quat(Arm_init_G[:, i])
        x[6:9, i] = Arm_Quat_D[1:]
        x[12, i] = Arm_Quat_D[0]
        x[9:12, i] = Arm_Quat_G[1:]
        x[13, i] = Arm_Quat_G[0]
    x_init = InitialGuessList()
    x_init.add(x, interpolation=InterpolationType.LINEAR)

    u_init = InitialGuessList()
    u_init.add([tau_init] * n_tau)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        ode_solver=ode_solver,
    )


def main():
    """
    Prepares and solves an ocp that has quaternion in it. Animates the results
    """

    ocp = prepare_ocp(
        "TruncAnd2Arm_Quaternion.bioMod",
        n_shooting=5,
        final_time=0.25,
    )
    sol = ocp.solve(show_online_optim=True)
    print("\n")

    # Print the last solution
    sol.animate()


if __name__ == "__main__":
    main()
