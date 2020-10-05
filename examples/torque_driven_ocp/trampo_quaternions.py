import numpy as np
import biorbd

from bioptim import (
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    ObjectiveList,
    Objective,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    ShowResult,
    InterpolationType,
)


def eul2quat(eul):
    # xyz convention
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


def prepare_ocp(biorbd_model_path, number_shooting_points, final_time):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = nqdot  # biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Mayer.MINIMIZE_MARKERS, markers_idx=1, weight=-1)
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([[tau_min] * ntau, [tau_max] * ntau])

    # Initial guesses
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
    u_init.add([tau_init] * ntau)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
    )


if __name__ == "__main__":

    # changer le path quand ce sera pret
    ocp = prepare_ocp(
        "TruncAnd2Arm_Quaternion.bioMod",
        number_shooting_points=5,
        final_time=0.25,
    )
    sol = ocp.solve()
    print("\n")

    # Print the last solution
    result_plot = ShowResult(ocp, sol)
    result_plot.graphs()
