"""
This example is adapted from arm_reaching_muscle_driven.py to make it torque driven.
The states dynamics and stochastic dynamics are implemented implicitly by integrating with direct collocations (which
commits less integration errors than using a trapezoidal scheme and is closer to the implementation suggested in Gillis
2013 insuring that the covariance matrix always stays positive semi-definite).
"""

import pickle
import casadi as cas
import numpy as np

from bioptim import (
    StochasticOptimalControlProgram,
    ObjectiveFcn,
    Solver,
    StochasticBiorbdModel,
    ObjectiveList,
    NonLinearProgram,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    InterpolationType,
    SocpType,
    Node,
    ConstraintList,
    ConstraintFcn,
    InitialGuessList,
    ControlType,
    Axis,
)

from bioptim.examples.stochastic_optimal_control.arm_reaching_torque_driven_implicit import ExampleType


def sensory_reference(
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    stochastic_variables: cas.MX | cas.SX,
    nlp: NonLinearProgram,
):
    """
    This functions returns the sensory reference for the feedback gains.
    """
    q = states[nlp.states["q"].index]
    qdot = states[nlp.states["qdot"].index]
    hand_pos = nlp.model.markers(q)[2][:2]
    hand_vel = nlp.model.marker_velocities(q, qdot)[2][:2]
    hand_pos_velo = cas.vertcat(hand_pos, hand_vel)
    return hand_pos_velo


def prepare_socp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    hand_final_position: np.ndarray,
    motor_noise_magnitude: cas.DM,
    sensory_noise_magnitude: cas.DM,
    example_type=ExampleType.CIRCLE,
) -> StochasticOptimalControlProgram:
    """
    The initialization of an ocp
    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    hand_final_position: np.ndarray
        The final position of the end effector
    motor_noise_magnitude: cas.DM
        The magnitude of the motor noise
    sensory_noise_magnitude: cas.DM
        The magnitude of the sensory noise
    example_type
        The type of problem to solve (CIRCLE or BAR)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    problem_type = SocpType.COLLOCATION(polynomial_degree=3, method="legendre")

    bio_model = StochasticBiorbdModel(
        biorbd_model_path,
        sensory_noise_magnitude=sensory_noise_magnitude,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_reference=sensory_reference,
        n_references=4,  # This number must be in agreement with what is declared in sensory_reference
        n_noised_states=4,
        n_noised_controls=2,
        n_collocation_points=3 + 1,
        friction_coefficients=np.array([[0.05, 0.025], [0.025, 0.05]]),
    )

    n_tau = bio_model.nb_tau
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    n_states = n_q * 2

    shoulder_pos_initial = 0.349065850398866
    shoulder_pos_final = 0.959931088596881
    elbow_pos_initial = 2.245867726451909  # Optimized in Tom's version
    elbow_pos_final = 1.159394851847144  # Optimized in Tom's version

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL, key="tau", weight=1e3 / 2, quadratic=True
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.STOCHASTIC_MINIMIZE_EXPECTED_FEEDBACK_EFFORTS,
        node=Node.ALL,
        weight=1e3 / 2,
        quadratic=False,
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_STATE, key="q", node=Node.START, target=np.array([shoulder_pos_initial, elbow_pos_initial])
    )
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.START, target=np.array([0, 0]))
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.END, target=np.array([0, 0]))
    constraints.add(
        ConstraintFcn.TRACK_STATE, key="q", node=Node.ALL, min_bound=0, max_bound=180
    )  # This is a bug, it should be in radians

    # This constraint insures that the hand reaches the target with x_mean
    constraints.add(
        ConstraintFcn.TRACK_MARKERS, node=Node.END, target=hand_final_position, marker_index=2, axes=[Axis.X, Axis.Y]
        #todo: @pariterre why axes is plural while all others are singular?
    )
    # While this constraint insures that the hand still reaches the target with the proper position and velocity even
    # in the presence of noise
    if example_type == ExampleType.BAR:
        max_bounds_lateral_variation = cas.inf
    elif example_type == ExampleType.CIRCLE:
        max_bounds_lateral_variation = 0.004
    else:
        raise NotImplementedError("Wrong problem type")

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        node=Node.END,
        marker_index=2,
        axes=[Axis.X, Axis.Y],
        min_bound=np.array([-cas.inf, -cas.inf]),
        max_bound=np.array([max_bounds_lateral_variation**2, 0.004**2]),
        is_stochastic=True,
    )
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        node=Node.END,
        marker_index=2,
        axes=[Axis.X, Axis.Y],
        min_bound=np.array([-cas.inf, -cas.inf]),
        max_bound=np.array([0.05**2, 0.05**2]),
        is_stochastic=True,
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.STOCHASTIC_TORQUE_DRIVEN,
        problem_type=problem_type,
        with_cholesky=False,
        expand_dynamics=False,
    )

    x_bounds = BoundsList()
    x_bounds.add("q", min_bound=[-cas.inf] * n_q, max_bound=[cas.inf] * n_q, interpolation=InterpolationType.CONSTANT)
    x_bounds.add(
        "qdot",
        min_bound=[-cas.inf] * n_qdot,
        max_bound=[cas.inf] * n_qdot,
        interpolation=InterpolationType.CONSTANT,
    )

    u_bounds = BoundsList()
    u_bounds.add(
        "tau", min_bound=[-cas.inf] * n_tau, max_bound=[cas.inf] * n_tau, interpolation=InterpolationType.CONSTANT
    )

    # Initial guesses
    states_init = np.zeros((n_states, n_shooting + 1))
    states_init[0, :] = np.linspace(shoulder_pos_initial, shoulder_pos_final, n_shooting + 1)
    states_init[1, :] = np.linspace(elbow_pos_initial, elbow_pos_final, n_shooting + 1)
    states_init[n_states:, :] = 0.01

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=states_init[:n_q, :], interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", initial_guess=states_init[n_q : n_q + n_qdot, :], interpolation=InterpolationType.EACH_FRAME)

    controls_init = np.ones((n_tau, n_shooting + 1)) * 0.01

    u_init = InitialGuessList()
    u_init.add("tau", initial_guess=controls_init, interpolation=InterpolationType.EACH_FRAME)

    s_init = InitialGuessList()
    s_bounds = BoundsList()
    n_k = 2 * 4
    n_ref = 4
    n_m = 4 * 4 * (3 + 1)
    n_cov = 4 * 4

    s_init.add("k", initial_guess=[0.01] * n_k, interpolation=InterpolationType.CONSTANT)
    s_bounds.add(
        "k",
        min_bound=[-cas.inf] * n_k,
        max_bound=[cas.inf] * n_k,
        interpolation=InterpolationType.CONSTANT,
    )

    s_init.add(
        "ref",
        initial_guess=[0.01] * n_ref,
        interpolation=InterpolationType.CONSTANT,
    )
    s_bounds.add(
        "ref",
        min_bound=[-cas.inf] * n_ref,
        max_bound=[cas.inf] * n_ref,
        interpolation=InterpolationType.CONSTANT,
    )

    s_init.add(
        "m",
        initial_guess=[0.01] * n_m,
        interpolation=InterpolationType.CONSTANT,
    )
    s_bounds.add(
        "m",
        min_bound=[-cas.inf] * n_m,
        max_bound=[cas.inf] * n_m,
        interpolation=InterpolationType.CONSTANT,
    )

    cov_init = cas.DM_eye(n_states) * np.array([1e-4, 1e-4, 1e-7, 1e-7])
    idx = 0
    cov_init_vector = np.zeros((n_states * n_states, 1))
    for i in range(n_states):
        for j in range(n_states):
            cov_init_vector[idx] = cov_init[i, j]
    s_init.add(
        "cov",
        initial_guess=cov_init_vector,
        interpolation=InterpolationType.CONSTANT,
    )
    s_bounds.add(
        "cov",
        min_bound=[-cas.inf] * n_cov,
        max_bound=[cas.inf] * n_cov,
        interpolation=InterpolationType.CONSTANT,
    )

    return StochasticOptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        s_init=s_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        s_bounds=s_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        control_type=ControlType.CONSTANT_WITH_LAST_NODE,
        n_threads=2,
        problem_type=problem_type,
    )


def main():
    # --- Options --- #
    vizualize_sol_flag = True

    biorbd_model_path = "models/LeuvenArmModel.bioMod"

    hand_final_position = np.array([9.359873986980460e-12, 0.527332023564034])  # Directly from Tom's version

    # --- Prepare the ocp --- #
    dt = 0.01
    final_time = 0.8
    n_shooting = int(final_time / dt)

    # --- Noise constants --- #
    motor_noise_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024

    motor_noise_magnitude = cas.DM(np.array([motor_noise_std**2 / dt, motor_noise_std**2 / dt]))
    wPq_magnitude = cas.DM(np.array([wPq_std**2 / dt, wPq_std**2 / dt]))
    wPqdot_magnitude = cas.DM(np.array([wPqdot_std**2 / dt, wPqdot_std**2 / dt]))
    sensory_noise_magnitude = cas.vertcat(wPq_magnitude, wPqdot_magnitude)

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_linear_solver("mumps")
    solver.set_tol(1e-3)
    solver.set_dual_inf_tol(3e-4)
    solver.set_constr_viol_tol(1e-7)
    solver.set_maximum_iterations(10000)
    solver.set_hessian_approximation("limited-memory")
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    solver.set_nlp_scaling_method("none")

    example_type = ExampleType.CIRCLE
    force_field_magnitude = 0
    socp = prepare_socp(
        biorbd_model_path=biorbd_model_path,
        final_time=final_time,
        n_shooting=n_shooting,
        hand_final_position=hand_final_position,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        example_type=example_type,
    )

    sol_socp = socp.solve(solver)
    # sol_socp.graphs()

    q_sol = sol_socp.states["q"]
    qdot_sol = sol_socp.states["qdot"]
    tau_sol = sol_socp.controls["tau"]
    k_sol = sol_socp.stochastic_variables["k"]
    ref_sol = sol_socp.stochastic_variables["ref"]
    m_sol = sol_socp.stochastic_variables["m"]
    cov_sol = sol_socp.stochastic_variables["cov"]
    a_sol = sol_socp.stochastic_variables["a"]
    c_sol = sol_socp.stochastic_variables["c"]
    stochastic_variables_sol = np.vstack((k_sol, ref_sol, m_sol, cov_sol, a_sol, c_sol))
    data = {
        "q_sol": q_sol,
        "qdot_sol": qdot_sol,
        "tau_sol": tau_sol,
        "k_sol": k_sol,
        "ref_sol": ref_sol,
        "m_sol": m_sol,
        "cov_sol": cov_sol,
        "a_sol": a_sol,
        "c_sol": c_sol,
        "stochastic_variables_sol": stochastic_variables_sol,
    }

    # --- Save the results --- #
    with open(f"leuvenarm_torque_driven_socp_{str(example_type)}_forcefield{force_field_magnitude}.pkl", "wb") as file:
        pickle.dump(data, file)

    # --- Visualize the results --- #
    if vizualize_sol_flag:
        import bioviz

        b = bioviz.Viz(model_path=biorbd_model_path)
        b.load_movement(q_sol)
        b.exec()


if __name__ == "__main__":
    main()
