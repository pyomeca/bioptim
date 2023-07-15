
"""
The goal of this program is to optimize the movement to achieve a 42/.
Phase 0 : Twist
Phase 1 : preparation for landing
"""

import numpy as np
import pickle
import biorbd_casadi as biorbd
import casadi as cas
import time
import sys

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    InterpolationType,
    OdeSolver,
    Node,
    Solver,
    BiMappingList,
    CostType,
    ConstraintList,
    ConstraintFcn,
    PenaltyController,
    BiorbdModel,
    Shooting,
    SolutionIntegrator,
)

def custom_trampoline_bed_in_peripheral_vision(controller: PenaltyController) -> cas.MX:
    """
    This function aims to encourage the avatar to keep the trampoline bed in his peripheral vision.
    It is done by discretizing the vision cone into vectors and determining if the vector projection of the gaze are inside the trampoline bed.
    """

    q = controller
    a = 1.07  # Trampoline with/2
    b = 2.14  # Trampoline length/2
    n = 6  # order of the polynomial for the trampoline bed rectangle equation

    # Get the gaze vector
    eyes_vect_start_marker_idx = controller.model.marker_index(f'eyes_vect_start')
    eyes_vect_end_marker_idx = controller.model.marker_index(f'eyes_vect_end')
    gaze_vector = controller.model.markers(controller.states["q"].mx)[eyes_vect_end_marker_idx] - controller.model.markers(controller.states["q"].mx)[eyes_vect_start_marker_idx]

    point_in_the_plane = np.array([1, 2, -0.83])
    vector_normal_to_the_plane = np.array([0, 0, 1])
    obj = 0
    for i_r in range(11):
        for i_th in range(10):

            # Get this vector from the vision cone
            marker_idx = controller.model.marker_index(f'cone_approx_{i_r}_{i_th}')
            vector_origin = controller.model.markers(controller.states["q"].mx)[eyes_vect_start_marker_idx]
            vector_end = controller.model.markers(controller.states["q"].mx)[marker_idx]
            vector = vector_end - vector_origin

            # Get the intersection between the vector and the trampoline plane
            t = (cas.dot(point_in_the_plane, vector_normal_to_the_plane) - cas.dot(vector_normal_to_the_plane, vector_origin)) / cas.dot(
                vector, vector_normal_to_the_plane
            )
            point_projection = vector_origin + vector * cas.fabs(t)

            # Determine if the point is inside the trampoline bed
            # Rectangle equation : (x/a)**n + (y/b)**n = 1
            # The function is convoluted with tanh to make it:
            # 1. Continuous
            # 2. Not encourage to look to the middle of the trampoline bed
            # 3. Largely penalized when outside the trampoline bed
            # 4. Equaly penalized when looking upward
            obj += cas.tanh(((point_projection[0]/a)**n + (point_projection[1]/b)**n) - 1) + 1

    val = cas.if_else(gaze_vector[2] > -0.01, 2*10*11,
                cas.if_else(cas.fabs(gaze_vector[0]/gaze_vector[2]) > np.tan(3*np.pi/8), 2*10*11,
                            cas.if_else(cas.fabs(gaze_vector[1]/gaze_vector[2]) > np.tan(3*np.pi/8), 2*10*11, obj)))
    out = controller.mx_to_cx("peripheral_vision", val, controller.states["q"])

    return out


def prepare_ocp(
    biorbd_model_path: str, n_shooting: tuple, num_twists: int, n_threads: int, ode_solver: OdeSolver = OdeSolver.RK4(), WITH_VISUAL_CRITERIA: bool = False
) -> OptimalControlProgram:
    """
    Prepare the ocp
    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod file
    n_shooting: int
        The number of shooting points
    ode_solver: OdeSolver
        The ode solver to use
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    final_time = 1.47
    biorbd_model = (
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
    )

    nb_q = biorbd_model[0].nb_q
    nb_qdot = biorbd_model[0].nb_qdot
    nb_qddot_joints = nb_q - biorbd_model[0].nb_root


    # for lisibility
    if not WITH_VISUAL_CRITERIA:
        X = 0
        Y = 1
        Z = 2
        Xrot = 3
        Yrot = 4
        Zrot = 5
        ZrotRightUpperArm = 6
        YrotRightUpperArm = 7
        ZrotLeftUpperArm = 8
        YrotLeftUpperArm = 9
        vX = 0 + nb_q
        vY = 1 + nb_q
        vZ = 2 + nb_q
        vXrot = 3 + nb_q
        vYrot = 4 + nb_q
        vZrot = 5 + nb_q
        vZrotRightUpperArm = 6 + nb_q
        vYrotRightUpperArm = 7 + nb_q
        vZrotLeftUpperArm = 8 + nb_q
        vYrotLeftUpperArm = 9 + nb_q
    else:
        X = 0
        Y = 1
        Z = 2
        Xrot = 3
        Yrot = 4
        Zrot = 5
        ZrotHead = 6
        XrotHead = 7
        ZrotEyes = 8
        XrotEyes = 9
        ZrotRightUpperArm = 10
        YrotRightUpperArm = 11
        ZrotLeftUpperArm = 12
        YrotLeftUpperArm = 13
        vX = 0 + nb_q
        vY = 1 + nb_q
        vZ = 2 + nb_q
        vXrot = 3 + nb_q
        vYrot = 4 + nb_q
        vZrot = 5 + nb_q
        vZrotHead = 6 + nb_q
        vXrotHead = 7 + nb_q
        vZrotEyes = 8 + nb_q
        vXrotEyes = 9 + nb_q
        vZrotRightUpperArm = 10 + nb_q
        vYrotRightUpperArm = 11 + nb_q
        vZrotLeftUpperArm = 12 + nb_q
        vYrotLeftUpperArm = 13 + nb_q

    # Add objective functions
    objective_functions = ObjectiveList()

    # Min controls
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=0
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=1
    )

    # Min control derivative
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=0, derivative=True,
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=1, derivative=True,
    )

    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time, weight=0.00001, phase=0
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=0.00001, phase=1
    )

    # aligning with the FIG regulations
    objective_functions.add(
         ObjectiveFcn.Lagrange.MINIMIZE_STATE,
         key="q",
         node=Node.ALL_SHOOTING,
         index=[YrotRightUpperArm, YrotLeftUpperArm],
         weight=100,
         phase=0,
    )


    if WITH_VISUAL_CRITERIA:

        # Spotting
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_SEGMENT_VELOCITY, segment="Head", weight=10000, phase=1)

        # Self-motion detection
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key='qdot', index=[ZrotEyes, XrotEyes], weight=100, phase=0)

        # Keeping the trampoline bed in the peripheral vision
        objective_functions.add(custom_trampoline_bed_in_peripheral_vision, custom_type=ObjectiveFcn.Lagrange, weight=100, phase=0)

        # Quiet eye
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS,
                                vector_0_marker_0="eyes_vect_start",
                                vector_0_marker_1="eyes_vect_end",
                                vector_1_marker_0="eyes_vect_start",
                                vector_1_marker_1="fixation_front",
                                weight=1, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS,
                                vector_0_marker_0="eyes_vect_start",
                                vector_0_marker_1="eyes_vect_end",
                                vector_1_marker_0="eyes_vect_start",
                                vector_1_marker_1="fixation_front",
                                weight=10000, phase=1)

        # Avoid extreme eye and neck angles
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[ZrotHead, XrotHead, ZrotEyes, XrotEyes], weight=10, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[ZrotHead, XrotHead, ZrotEyes, XrotEyes], weight=10, phase=1)


    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)

    u_bounds = BoundsList()
    u_init = InitialGuessList()

    x_bounds = BoundsList()
    x_init = InitialGuessList()

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        [final_time / len(biorbd_model)] * len(biorbd_model),
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        ode_solver=ode_solver,
        n_threads=n_threads,
        assume_phase_dynamics=True,
    )


def main():
    """
    Prepares and solves an ocp for a 42/ with and without visual criteria.
    """

    WITH_VISUAL_CRITERIA = True

    if WITH_VISUAL_CRITERIA:
        biorbd_model_path = "models/SoMe_42_with_visual_criteria.bioMod"
    else:
        raise RuntimeError("No need for this test")

    n_shooting = (100, 40)
    num_twists = 1
    ocp = prepare_ocp(biorbd_model_path, n_shooting=n_shooting, num_twists=num_twists, n_threads=7, WITH_VISUAL_CRITERIA=WITH_VISUAL_CRITERIA)
    # ocp.add_plot_penalty(CostType.ALL)

    # solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(10)
    solver.set_convergence_tolerance(1e-4)

    tic = time.time()
    sol = ocp.solve(solver)
    toc = time.time() - tic
    print(toc)  
    # Before: 3.6.2: 74.02791595458984s 
    # Before: 3.6.3: 81s

    timestamp = time.strftime("%Y-%m-%d-%H%M")
    name = biorbd_model_path.split("/")[-1].removesuffix(".bioMod")
    qs = sol.states[0]["q"]
    qdots = sol.states[0]["qdot"]
    qddots = sol.controls[0]["qddot_joints"]
    for i in range(1, len(sol.states)):
        qs = np.hstack((qs, sol.states[i]["q"]))
        qdots = np.hstack((qdots, sol.states[i]["qdot"]))
        qddots = np.hstack((qddots, sol.controls[i]["qddot_joints"]))
    time_parameters = sol.parameters["time"]


    integrated_sol = sol.integrate(shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.SCIPY_DOP853)

    time_vector = integrated_sol.time[0]
    q_reintegrated = integrated_sol.states[0]["q"]
    qdot_reintegrated = integrated_sol.states[0]["qdot"]
    for i in range(1, len(sol.states)):
        time_vector = np.hstack((time_vector, integrated_sol.time[i]))
        q_reintegrated = np.hstack((q_reintegrated, integrated_sol.states[i]["q"]))
        qdot_reintegrated = np.hstack((qdot_reintegrated, integrated_sol.states[i]["qdot"]))

    del sol.ocp
    with open(f"Solutions/{name}-{str(n_shooting).replace(', ', '_')}-{timestamp}.pkl", "wb") as f:
        pickle.dump((sol, qs, qdots, qddots, time_parameters, q_reintegrated, qdot_reintegrated, time_vector), f)

    # sol.animate(n_frames=-1, show_floor=False)

if __name__ == "__main__":
    main()