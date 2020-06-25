import numpy as np
import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    ProblemType,
    Objective,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    InterpolationType,
)

def eul2quat(eul):
    # xyz convention
    ph  = eul[0]
    th  = eul[1]
    ps  = eul[2]
    cph = np.cos(ph * 0.5)
    sph = np.sin(ph * 0.5)
    cth = np.cos(th * 0.5)
    sth = np.sin(th * 0.5)
    cps = np.cos(ps * 0.5)
    sps = np.sin(ps * 0.5)
    w = -sph * sth * sps + cph * cth * cps
    x =  sph * cth * cps + cph * sth * sps
    y =  cph * sth * cps - sph * cth * sps
    z =  sph * sth * cps + cph * cth * sps
    return np.array([w,x,y,z])

def prepare_ocp(biorbd_model_path, number_shooting_points, final_time):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = nqdot # biorbd_model.nbGeneralizedTorque()
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = ({"type": Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT},{"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100})

    # Dynamics
    problem_type = {"type": ProblemType.TORQUE_DRIVEN}

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    # Define control path constraint
    U_bounds = Bounds([torque_min] * ntau, [torque_max] * ntau)

    # Initial guesses
    x = np.vstack((np.zeros((biorbd_model.nbQ(),2)), np.ones((biorbd_model.nbQdot(),2))))
    Arm_init = np.zeros((3, 2))
    Arm_init[1, 0] = 0
    Arm_init[1, 1] = -np.pi
    for i in range(2):
        Arm_Quat = eul2quat(Arm_init[:,i])
        # Arm_Quat = biorbd.Quaternion_fromMatrix(biorbd.Rotation_fromEulerAngles(biorbd.Vector3d(Arm_init[0,i], Arm_init[1,i], Arm_init[2,i]),'xyz')).to_mx() # not the right type
        x[6:9, i] = Arm_Quat[1:]
        x[9, i] = Arm_Quat[0]
    X_init = InitialConditions(x, interpolation_type=InterpolationType.LINEAR)

    U_init = InitialConditions([torque_init] * ntau)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
    )


if __name__ == "__main__":

# changer le path quand ce sera pret
    ocp = prepare_ocp("/home/user/Documents/Programmation/BiorbdOptim/examples/sandbox/TruncAndArm_Quaternion.bioMod", number_shooting_points=5, final_time=0.25)
    sol = ocp.solve()
    print("\n")

    # Print the last solution
    result_plot = ShowResult(ocp, sol)
    result_plot.graphs()
