"""
This example shows how to reconstruct a walking movement to match as closely as possible a participant's behavior.
Please note that, in this example, the forces measured by the force plates are applied directly to the model's feet,
meaning that contacts between the participant and the treadmill are not modeled. Moreover, residual forces applied on
the feet were added as controls to help convergence and mitigate the effect of noise in the force platform data.
"""
import pickle
import numpy as np

from bioptim import (
    InitialGuessList,
    InterpolationType,
    ObjectiveFcn,
    ObjectiveList,
    OptimalControlProgram,
    PhaseDynamics,
    BoundsList,
    Solver,
    OdeSolver,
    ExternalForceSetTimeSeries,
    Node,
    DynamicsOptionsList,
    OdeSolverBase,
    SolutionMerge,
    TimeAlignment,
)

from model import WithResidualExternalForces, animate_solution


def prepare_ocp(
    biorbd_model_path: str,
    mesh_file_folder: str,
    n_shooting: int,
    phase_time: float,
    q_exp: np.ndarray[float],
    qdot_exp: np.ndarray[float],
    tau_exp: np.ndarray[float],
    f_ext_exp: dict[str, np.ndarray[float]],
    emg_normalized_exp: np.ndarray[float],
    markers_exp: np.ndarray[float],
    ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=5),
    n_threads: int = 8,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path the biorbd model (.bioMod)
    n_shooting: int
        The number of shooting nodes (here, also the number of experimental frames in the cycle)
    phase_time: float
        The time of the phase (here, the whole duration of the cycle)
    q_exp: np.ndarray[float]
        The experimental joint angles from inverse kinematics (shape: nb_q x n_shooting + 1)
    qdot_exp: np.ndarray[float]
        The experimental joint velocities from finite difference (shape: nb_q x n_shooting + 1)
    tau_exp: np.ndarray[float]
        The experimental joint torques from inverse dynamics (shape: nb_q x n_shooting + 1)
    f_ext_exp: dict[str, np.ndarray[float]]
        The experimental external forces (shape: 9 [Px, Py, Pz, Mx, My, Mz, Fx, Fy, Fz] x n_shooting)
    emg_normalized_exp: np.ndarray[float]
        The experimental EMG signals normalized by MVC (shape: nb_muscles x n_shooting + 1)
    markers_exp: np.ndarray[float]
        The experimental markers (shape: nb_markers x 3 x n_shooting + 1)
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    n_threads: int
        Number of threads to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # External force set
    external_force_set = ExternalForceSetTimeSeries(nb_frames=n_shooting)
    external_force_set.add(
        force_name="calcn_l",
        segment="calcn_l",
        values=f_ext_exp["left_leg"][3:9, :-1],
        point_of_application=f_ext_exp["left_leg"][:3, :-1],
    )
    external_force_set.add(
        force_name="calcn_r",
        segment="calcn_r",
        values=f_ext_exp["right_leg"][3:9, :-1],
        point_of_application=f_ext_exp["right_leg"][:3, :-1],
    )
    numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}

    # Model
    bio_model = WithResidualExternalForces(biorbd_model_path,
                                            mesh_file_folder=mesh_file_folder,
                                            external_force_set=external_force_set)

    nb_q = bio_model.nb_q
    nb_muscles = bio_model.nb_muscles
    r_foot_marker_index = np.array(
        [
            bio_model.marker_index(f"RCAL"),
            bio_model.marker_index(f"RMFH1"),
            bio_model.marker_index(f"RMFH5"),
            bio_model.marker_index(f"R_foot_up"),
        ]
    )
    l_foot_marker_index = np.array(
        [
            bio_model.marker_index(f"LCAL"),
            bio_model.marker_index(f"LMFH1"),
            bio_model.marker_index(f"LMFH5"),
            bio_model.marker_index(f"L_foot_up"),
        ]
    )

    # Declaration of the objectives
    objective_functions = ObjectiveList()
    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="tau",
        weight=0.001,
    )
    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="tau",
        weight=0.1,
        index=[0, 1, 2, 3, 4, 5],
    )
    # Note: all muscles have a target except tfl, which did not have an EMG
    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="muscles",
        weight=10,
        target=emg_normalized_exp[:, :-1],
    )
    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=100.0, node=Node.ALL, target=markers_exp
    )
    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.TRACK_MARKERS,
        weight=1000.0,
        node=Node.ALL,
        marker_index=["RCAL", "RMFH1", "RMFH5", "R_foot_up", "LCAL", "LMFH1", "LMFH5", "L_foot_up"],
        target=markers_exp[:, np.hstack((r_foot_marker_index, l_foot_marker_index)), :],
    )
    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=1.0, node=Node.ALL, target=q_exp
    )
    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.TRACK_STATE,
        key="qdot",
        node=Node.ALL,
        weight=0.01,
        target=qdot_exp,
    )
    objective_functions.add(  # Minimize residual contact forces
        objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="contact_forces",
        node=Node.ALL_SHOOTING,
        weight=10,
    )
    objective_functions.add(  # Track CoP position
        objective=ObjectiveFcn.Lagrange.TRACK_CONTROL,
        key="contact_positions",
        node=Node.ALL_SHOOTING,
        weight=0.01,
        target=np.vstack((f_ext_exp["left_leg"][0:3, :-1], f_ext_exp["right_leg"][0:3, :-1])),
    )

     # No constraints

    dynamics = DynamicsOptionsList()
    dynamics.add(
        numerical_data_timeseries=numerical_time_series,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        ode_solver=ode_solver,
    )

    x_bounds = BoundsList()
    # Bounds personalized to the subject's current kinematics
    min_q = q_exp[:, :] - 0.3
    min_q[:6, :] = q_exp[:6, :] - 0.05
    max_q = q_exp[:, :] + 0.3
    max_q[:6, :] = q_exp[:6, :] + 0.05
    x_bounds.add("q", min_bound=min_q, max_bound=max_q, interpolation=InterpolationType.EACH_FRAME)
    # Bounds personalized to the subject's current joint velocities (not a real limitation, so it is executed with +-10)
    x_bounds.add(
        "qdot",
        min_bound=qdot_exp - 10,
        max_bound=qdot_exp + 10,
        interpolation=InterpolationType.EACH_FRAME,
    )

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=q_exp, interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", initial_guess=qdot_exp, interpolation=InterpolationType.EACH_FRAME)

    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[-800] * nb_q, max_bound=[800] * nb_q, interpolation=InterpolationType.CONSTANT)
    u_bounds.add(
        "muscles",
        min_bound=[0.0001] * nb_muscles,
        max_bound=[1.0] * nb_muscles,
        interpolation=InterpolationType.CONSTANT,
    )
    u_bounds.add(
        "contact_forces", min_bound=[-100] * 6, max_bound=[100] * 6, interpolation=InterpolationType.CONSTANT
    )
    u_bounds.add(
        "contact_positions", min_bound=[-2, -2, 0.0, -2, -2, 0.0], max_bound=[2, 2, 0.005, 2, 2, 0.005], interpolation=InterpolationType.CONSTANT
    )

    u_init = InitialGuessList()
    u_init.add("tau", initial_guess=tau_exp[:, :-1], interpolation=InterpolationType.EACH_FRAME)
    u_init.add(
        "muscles", initial_guess=emg_normalized_exp[:, :-1], interpolation=InterpolationType.EACH_FRAME
    )
    u_init.add("contact_forces", initial_guess=[0] * 6, interpolation=InterpolationType.CONSTANT)
    u_init.add(
        "contact_positions",
        initial_guess=np.vstack(
            (f_ext_exp["left_leg"][0:3, :-1], f_ext_exp["right_leg"][0:3, :-1])
        ),
        interpolation=InterpolationType.EACH_FRAME,
    )

    return OptimalControlProgram(
        bio_model=bio_model,
        n_shooting=n_shooting,
        phase_time=phase_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        use_sx=False,
        n_threads=n_threads,
    )


def main():

    # Get the experimental data
    with open("opc_data.pkl", "rb") as f:
        data = pickle.load(f)
        n_shooting = data["n_shooting"]
        phase_time = data["phase_time"]
        q_exp = data["q_exp"]
        qdot_exp = data["qdot_exp"]
        tau_exp = data["tau_exp"]
        f_ext_exp = data["f_ext_exp"]
        emg_normalized_exp = data["emg_normalized_exp"]
        markers_exp = data["markers_exp"]

    # --- Prepare the ocp --- #
    biorbd_model_path = "../../models/wholebody_model.bioMod"
    mesh_file_folder = "../../../external/biomechanics_models/Geometry_triangles"
    ocp = prepare_ocp(
        biorbd_model_path,
        mesh_file_folder,
        n_shooting,
        phase_time,
        q_exp,
        qdot_exp,
        tau_exp,
        f_ext_exp,
        emg_normalized_exp,
        markers_exp,
    )
    ocp.add_plot_penalty()
    ocp.add_plot_ipopt_outputs()

    # --- Solve the ocp --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    # solver.set_linear_solver("ma57")  # This is recommended
    solver.set_maximum_iterations(1000)
    solver.set_tol(1e-6)
    sol = ocp.solve(solver=solver)

    # Get the optimal solution
    time_opt = sol.decision_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
    q_opt = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
    qdot_opt = sol.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
    tau_opt = sol.decision_controls(to_merge=SolutionMerge.NODES)["tau"]
    muscles_opt = sol.decision_controls(to_merge=SolutionMerge.NODES)["muscles"]
    f_ext_value_opt = sol.decision_controls(to_merge=SolutionMerge.NODES)["contact_forces"]
    f_ext_position_opt = sol.decision_controls(to_merge=SolutionMerge.NODES)["contact_positions"]

    # --- Animation --- #
    animate_solution(biorbd_model_path,
                     float(time_opt),
                     n_shooting,
                    markers_exp,
                    f_ext_exp,
                    q_opt,
                    muscles_opt,
                    f_ext_position_opt,
                    f_ext_value_opt,
                    )


if __name__ == "__main__":
    main()
