"""
This example is adapted from arm_reaching_muscle_driven.py to make it torque driven.
The states dynamics is explicit, while the stochastic variables dynamics is implicit.
This formulation allow to decouple the covariance matrix with the previous states reducing the complexity of resolution,
but increases largely the number of variables to optimize.
Decomposing the covariance matrix using Cholesky L @ @.T allows to reduce the number of variables and ensures that the
covariance matrix always stays positive semi-definite.
"""

import platform

import pickle
import casadi as cas
import numpy as np
from enum import Enum

from bioptim import (
    OptimalControlProgram,
    StochasticOptimalControlProgram,
    ObjectiveFcn,
    Solver,
    BiorbdModel,
    ObjectiveList,
    NonLinearProgram,
    DynamicsEvaluation,
    DynamicsFunctions,
    ConfigureProblem,
    DynamicsList,
    BoundsList,
    InterpolationType,
    SocpType,
    PenaltyController,
    Node,
    ConstraintList,
    ConstraintFcn,
    MultinodeConstraintList,
    InitialGuessList,
    Axis,
    DynamicsFcn,
    OdeSolver,
    ControlType,
    VariableScalingList,
)


class ExampleType(Enum):
    """
    Selection of the type of example to solve
    """

    CIRCLE = "CIRCLE"
    BAR = "BAR"


def get_force_field(q, force_field_magnitude):
    """
    Get the effect of the force field.

    Parameters
    ----------
    q: MX.sym
        The generalized coordinates
    force_field_magnitude: float
        The magnitude of the force field
    """
    l1 = 0.3
    l2 = 0.33
    f_force_field = force_field_magnitude * (l1 * cas.cos(q[0]) + l2 * cas.cos(q[0] + q[1]))
    hand_pos = cas.MX(2, 1)
    hand_pos[0] = l2 * cas.sin(q[0] + q[1]) + l1 * cas.sin(q[0])
    hand_pos[1] = l2 * cas.sin(q[0] + q[1])
    tau_force_field = -f_force_field @ hand_pos
    return tau_force_field

def sensory_reference_function(states: cas.MX | cas.SX,
                          controls: cas.MX | cas.SX,
                          parameters: cas.MX | cas.SX,
                          stochastic_variables: cas.MX | cas.SX,
                          nlp: NonLinearProgram):
    """
    This functions returns the sensory reference for the feedback gains.
    """
    q = states[nlp.states["q"].index]
    qdot = states[nlp.states["qdot"].index]
    hand_pos = nlp.model.markers(q)[2][:2]
    hand_vel = nlp.model.marker_velocities(q, qdot)[2][:2]
    hand_pos_velo = cas.vertcat(hand_pos, hand_vel)
    return hand_pos_velo

def stochastic_forward_dynamics(
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    stochastic_variables: cas.MX | cas.SX,
    nlp: NonLinearProgram,
    motor_noise,
    sensory_noise,
    force_field_magnitude,
    with_gains,
) -> DynamicsEvaluation:
    """
    The dynamic function of the states including feedback gains.

    Parameters
    ----------
    states: MX.sym
        The states
    controls: MX.sym
        The controls
    parameters: MX.sym
        The parameters
    stochastic_variables: MX.sym
        The stochastic variables
    nlp: NonLinearProgram
        The current non-linear program
    motor_noise: MX.sym
        The motor noise
    sensory_noise: MX.sym
        The sensory noise
    force_field_magnitude: float
        The magnitude of the force field
    with_gains: bool
        If the feedback gains are included or not to the torques
    """
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    tau_fb = tau
    if with_gains:
        ref = DynamicsFunctions.get(nlp.stochastic_variables["ref"], stochastic_variables)
        k = DynamicsFunctions.get(nlp.stochastic_variables["k"], stochastic_variables)
        k_matrix = nlp.stochastic_variables["k"].reshape_sym_to_matrix(k)

        hand_pos_velo = nlp.model.sensory_reference_function(states, controls, parameters, stochastic_variables, nlp)

        tau_fb += k_matrix @ ((hand_pos_velo - ref) + sensory_noise)

    tau_force_field = nlp.model.force_field(q, force_field_magnitude)

    torques_computed = tau_fb + motor_noise + tau_force_field

    friction = nlp.model.friction_coefficients

    mass_matrix = nlp.model.mass_matrix(q)
    non_linear_effects = nlp.model.non_linear_effects(q, qdot)

    dqdot_computed = cas.inv(mass_matrix) @ (torques_computed - non_linear_effects - friction @ qdot)

    return DynamicsEvaluation(dxdt=cas.vertcat(qdot, dqdot_computed))


def configure_stochastic_optimal_control_problem(
    ocp: OptimalControlProgram, nlp: NonLinearProgram, motor_noise, sensory_noise, with_cholesky
):
    """
    Configure the stochastic optimal control problem.
    """
    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)
    ConfigureProblem.configure_tau(ocp, nlp, False, True)

    # Stochastic variables
    ConfigureProblem.configure_stochastic_k(ocp, nlp, n_noised_controls=2, n_feedbacks=4)
    ConfigureProblem.configure_stochastic_ref(ocp, nlp, n_references=4)
    ConfigureProblem.configure_stochastic_m(ocp, nlp, n_noised_states=4)
    if with_cholesky:
        ConfigureProblem.configure_stochastic_cholesky_cov(ocp, nlp, n_noised_states=4)
    else:
        ConfigureProblem.configure_stochastic_cov_implicit(ocp, nlp, n_noised_states=4)
    ConfigureProblem.configure_stochastic_a(ocp, nlp, n_noised_states=4)
    ConfigureProblem.configure_stochastic_c(ocp, nlp, n_feedbacks=4, n_noise=6)

    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda states, controls, parameters, stochastic_variables, nlp, motor_noise, sensory_noise: nlp.dynamics_type.dynamic_function(
            states, controls, parameters, stochastic_variables, nlp, motor_noise, sensory_noise, with_gains=False
        ),
        motor_noise=motor_noise,
        sensory_noise=sensory_noise,
    )
    ConfigureProblem.configure_stochastic_dynamics_function(
        ocp,
        nlp,
        noised_dyn_func=lambda states, controls, parameters, stochastic_variables, nlp, motor_noise, sensory_noise: nlp.dynamics_type.dynamic_function(
            states, controls, parameters, stochastic_variables, nlp, motor_noise, sensory_noise, with_gains=True
        ),
    )

def expected_feedback_effort(controller: PenaltyController, sensory_noise_magnitude: cas.DM) -> cas.MX:
    """
    This function computes the expected effort due to the motor command and feedback gains for a given sensory noise
    magnitude.
    It is computed as Jacobian(effort, states) @ cov @ Jacobian(effort, states) +
                        Jacobian(efforst, motor_noise) @ sigma_w @ Jacobian(efforst, motor_noise)

    Parameters
    ----------
    controller : PenaltyController
        Controller to be used to compute the expected effort.
    sensory_noise_magnitude : cas.DM
        Magnitude of the sensory noise.
    """
    n_tau = controller.controls["tau"].cx_start.shape[0]
    n_q = controller.states["q"].cx_start.shape[0]
    n_qdot = controller.states["qdot"].cx_start.shape[0]

    sensory_noise_matrix = sensory_noise_magnitude * cas.MX_eye(4)

    # create the casadi function to be evaluated
    # Get the symbolic variables
    ref = controller.stochastic_variables["ref"].cx_start
    if "cholesky_cov" in controller.stochastic_variables.keys():
        l_cov_matrix = controller.stochastic_variables["cholesky_cov"].reshape_to_cholesky_matrix(Node.START)
        cov_matrix = l_cov_matrix @ l_cov_matrix.T
    else:
        cov_matrix = controller.stochastic_variables["cov"].reshape_to_matrix(Node.START)

    k = controller.stochastic_variables["k"].cx_start
    k_matrix = cas.MX(n_q + n_qdot, n_tau)
    for s0 in range(n_q + n_qdot):
        for s1 in range(n_tau):
            k_matrix[s0, s1] = k[s0 * n_tau + s1]
    k_matrix = k_matrix.T

    # Compute the expected effort
    hand_pos_velo = controller.model.sensory_reference_function(controller.states.cx_start,
                                                                controller.controls.cx_start,
                                                                controller.parameters.cx_start,
                                                                controller.stochastic_variables.cx_start,
                                                                controller.get_nlp)
    trace_k_sensor_k = cas.trace(k_matrix @ sensory_noise_matrix @ k_matrix.T)
    e_fb = k_matrix @ ((hand_pos_velo - ref) + sensory_noise_magnitude)
    jac_e_fb_x = cas.jacobian(e_fb, controller.states.cx_start)
    trace_jac_p_jack = cas.trace(jac_e_fb_x @ cov_matrix @ jac_e_fb_x.T)
    expectedEffort_fb_mx = trace_jac_p_jack + trace_k_sensor_k
    func = cas.Function(
        "f_expectedEffort_fb",
        [controller.states.cx_start, controller.stochastic_variables.cx_start],
        [expectedEffort_fb_mx],
    )

    out = func(controller.states.cx_start, controller.stochastic_variables.cx_start)

    return out

def prepare_socp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ee_final_position: np.ndarray,
    motor_noise_magnitude: cas.DM,
    sensory_noise_magnitude: cas.DM,
    force_field_magnitude: float = 0,
    example_type=ExampleType.CIRCLE,
    with_cholesky: bool = False,
    with_scaling: bool = False,
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
    ee_final_position: np.ndarray
        The final position of the end effector
    motor_noise_magnitude: cas.DM
        The magnitude of the motor noise
    sensory_noise_magnitude: cas.DM
        The magnitude of the sensory noise
    force_field_magnitude: float
        The magnitude of the force field
    example_type
        The type of problem to solve (CIRCLE or BAR)
    with_cholesky: bool
        If True, whether to use the Cholesky factorization of the covariance matrix or not

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    problem_type = SocpType.SOCP_IMPLICIT(motor_noise_magnitude, sensory_noise_magnitude)

    bio_model = BiorbdModel(biorbd_model_path)
    bio_model.sensory_reference_function = sensory_reference_function
    bio_model.force_field = get_force_field
    bio_model.friction_coefficients = np.array([[0.05, 0.025], [0.025, 0.05]])

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

    objective_functions.add(ObjectiveFcn.Lagrange.STOCHASTIC_MINIMIZE_EXPECTED_FEEDBACK_EFFORTS,
                            sensory_noise_magnitude=sensory_noise_magnitude,
                            weight=1e3/2,
                            quadratic=True,
                            phase=0)

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
    constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.END, target=ee_final_position, marker_index=2,
                    axes=[Axis.X, Axis.Y])  ## merge conflict
    # While this constraint insures that the hand still reaches the target with the proper position and velocity even
    # in the presence of noise
    if example_type == ExampleType.BAR:
        max_bounds_lateral_variation = cas.inf
    elif example_type == ExampleType.CIRCLE:
        max_bounds_lateral_variation = 0.004
    else:
        raise NotImplementedError("Wrong problem type")
    constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.END, marker_index=2, axes=[Axis.X, Axis.Y],
                    phase=0, min_bound=np.array([-cas.inf, -cas.inf]),
                    max_bound=np.array([max_bounds_lateral_variation**2, 0.004**2]), is_stochastic=True,)  ## merge conflict
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, node=Node.END, marker_index=2, axes=[Axis.X, Axis.Y],
                    phase=0, min_bound=np.array([-cas.inf, -cas.inf]),
                    max_bound=np.array([0.05**2, 0.05**2]),
                    is_stochastic=True, )  ## merge conflict

    # Dynamics
    dynamics = DynamicsList()
    # dynamics.add(DynamicsFcn.STOCHASTIC_TORQUE_DRIVEN,
    #              expand=False,
    #              with_cholesky=with_cholesky,
    #              problem_type=problem_type,
    #              )
    # Fuck, merge with collocations needed to continue
    dynamics.add(
        configure_stochastic_optimal_control_problem,
        dynamic_function=lambda states, controls, parameters, stochastic_variables, nlp, motor_noise, sensory_noise, with_gains: stochastic_forward_dynamics(
            states,
            controls,
            parameters,
            stochastic_variables,
            nlp,
            motor_noise,
            sensory_noise,
            force_field_magnitude=force_field_magnitude,
            with_gains=with_gains,
        ),
        motor_noise=np.zeros((n_tau, 1)),
        sensory_noise=np.zeros((n_q + n_qdot, 1)),
        with_cholesky=with_cholesky,
        expand=False,
    )

    states_min = np.ones((n_states, n_shooting + 1)) * -cas.inf
    states_max = np.ones((n_states, n_shooting + 1)) * cas.inf

    x_bounds = BoundsList()
    x_bounds.add(
        "q", min_bound=states_min[:n_q, :], max_bound=states_max[:n_q, :], interpolation=InterpolationType.EACH_FRAME
    )
    x_bounds.add(
        "qdot",
        min_bound=states_min[n_q : n_q + n_qdot, :],
        max_bound=states_max[n_q : n_q + n_qdot, :],
        interpolation=InterpolationType.EACH_FRAME,
    )

    controls_min = np.ones((n_tau, 3)) * -cas.inf
    controls_max = np.ones((n_tau, 3)) * cas.inf

    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=controls_min, max_bound=controls_max)

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
    # K(2x4) + ref(4x1) + M(4x4)
    n_stochastic = n_tau * (n_q + n_qdot) + n_q + n_qdot + n_states * n_states
    if not with_cholesky:
        n_stochastic += n_states * n_states  # + cov(4, 4)
    else:
        n_cholesky_cov = 0
        for i in range(n_states):
            for j in range(i + 1):
                n_cholesky_cov += 1
        n_stochastic += n_cholesky_cov  # + cholesky_cov(10)
    stochastic_init = np.zeros((n_stochastic, n_shooting + 1))
    stochastic_min = np.ones((n_stochastic, 3)) * -cas.inf
    stochastic_max = np.ones((n_stochastic, 3)) * cas.inf
    curent_index = 0
    stochastic_init[: n_tau * (n_q + n_qdot), :] = 0.01  # K
    s_init.add(
        "k", initial_guess=stochastic_init[: n_tau * (n_q + n_qdot), :], interpolation=InterpolationType.EACH_FRAME
    )
    s_bounds.add(
        "k",
        min_bound=stochastic_min[: n_tau * (n_q + n_qdot), :],
        max_bound=stochastic_max[: n_tau * (n_q + n_qdot), :],
    )
    curent_index += n_tau * (n_q + n_qdot)
    stochastic_init[curent_index : curent_index + n_q + n_qdot, :] = 0.01  # ref
    s_init.add(
        "ref",
        initial_guess=stochastic_init[curent_index : curent_index + n_q + n_qdot, :],
        interpolation=InterpolationType.EACH_FRAME,
    )
    s_bounds.add(
        "ref",
        min_bound=stochastic_min[curent_index : curent_index + n_q + n_qdot, :],
        max_bound=stochastic_max[curent_index : curent_index + n_q + n_qdot, :],
    )
    curent_index += n_q + n_qdot
    stochastic_init[curent_index : curent_index + n_states * n_states, :] = 0.01  # M
    s_init.add(
        "m",
        initial_guess=stochastic_init[curent_index : curent_index + n_states * n_states, :],
        interpolation=InterpolationType.EACH_FRAME,
    )
    s_bounds.add(
        "m",
        min_bound=stochastic_min[curent_index : curent_index + n_states * n_states, :],
        max_bound=stochastic_max[curent_index : curent_index + n_states * n_states, :],
    )
    curent_index += n_states * n_states
    if not with_cholesky:
        stochastic_init[curent_index : curent_index + n_states * n_states, :] = 0.01  # cov
        cov_init = cas.DM_eye(n_states) * np.array([1e-4, 1e-4, 1e-7, 1e-7])
        idx = 0
        for i in range(n_states):
            for j in range(n_states):
                stochastic_init[idx, 0] = cov_init[i, j]
        s_init.add(
            "cov",
            initial_guess=stochastic_init[curent_index : curent_index + n_states * n_states, :],
            interpolation=InterpolationType.EACH_FRAME,
        )
        s_bounds.add(
            "cov",
            min_bound=stochastic_min[curent_index : curent_index + n_states * n_states, :],
            max_bound=stochastic_max[curent_index : curent_index + n_states * n_states, :],
        )
    else:
        stochastic_init[curent_index : curent_index + n_cholesky_cov, :] = 0.01  # cholsky cov
        cov_init = cas.DM_eye(n_states) * np.array([1e-4, 1e-4, 1e-7, 1e-7])
        idx = 0
        for i in range(n_states):
            for j in range(i + 1):
                stochastic_init[idx, 0] = cov_init[i, j]
        s_init.add(
            "cholesky_cov",
            initial_guess=stochastic_init[curent_index : curent_index + n_cholesky_cov, :],
            interpolation=InterpolationType.EACH_FRAME,
        )
        s_bounds.add(
            "cholesky_cov",
            min_bound=stochastic_min[curent_index : curent_index + n_cholesky_cov, :],
            max_bound=stochastic_max[curent_index : curent_index + n_cholesky_cov, :],
        )

    # Vaiables scaling
    u_scaling = VariableScalingList()
    if with_scaling:
        u_scaling["tau"] = [10] * n_tau

    # TODO: fix scaling is_symbolic
    s_scaling = VariableScalingList()
    if with_scaling:
        s_scaling["k"] = [100] * (n_tau * (n_q + n_qdot))
        s_scaling["ref"] = [1] * (n_q + n_qdot)
        s_scaling["m"] = [1] * (n_states * n_states)

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
        u_scaling=u_scaling,
        s_scaling=s_scaling,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=OdeSolver.TRAPEZOIDAL(),
        control_type=ControlType.CONSTANT_WITH_LAST_NODE,
        n_threads=1,
        assume_phase_dynamics=False,
        problem_type=problem_type,
    )


def main():
    # --- Options --- #
    vizualize_sol_flag = True
    with_cholesky = True
    with_scaling = False # True

    biorbd_model_path = "models/LeuvenArmModel.bioMod"

    ee_final_position = np.array([9.359873986980460e-12, 0.527332023564034])  # Directly from Tom's version

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
        ee_final_position=ee_final_position,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        example_type=example_type,
        force_field_magnitude=force_field_magnitude,
        with_cholesky=with_cholesky,
        with_scaling=with_scaling,
    )

    sol_socp = socp.solve(solver)
    # sol_socp.graphs()

    q_sol = sol_socp.states["q"]
    qdot_sol = sol_socp.states["qdot"]
    tau_sol = sol_socp.controls["tau"]
    k_sol = sol_socp.stochastic_variables["k"]
    ref_sol = sol_socp.stochastic_variables["ref"]
    m_sol = sol_socp.stochastic_variables["m"]
    if with_cholesky:
        cov_sol = None
        cholesky_cov_sol = sol_socp.stochastic_variables["cholesky_cov"]
    else:
        cov_sol = sol_socp.stochastic_variables["cov"]
        cholesky_cov_sol = None
    a_sol = sol_socp.stochastic_variables["a"]
    c_sol = sol_socp.stochastic_variables["c"]
    stochastic_variables_sol = np.vstack((k_sol, ref_sol, m_sol, cov_sol, cholesky_cov_sol, a_sol, c_sol))
    data = {
        "q_sol": q_sol,
        "qdot_sol": qdot_sol,
        "tau_sol": tau_sol,
        "k_sol": k_sol,
        "ref_sol": ref_sol,
        "m_sol": m_sol,
        "cov_sol": cov_sol,
        "cholesky_cov_sol": cholesky_cov_sol,
        "a_sol": a_sol,
        "c_sol": c_sol,
        "stochastic_variables_sol": stochastic_variables_sol,
    }

    # --- Save the results --- #
    with open(
        f"leuvenarm_torque_driven_socp_{str(example_type)}_forcefield{force_field_magnitude}_{with_cholesky}.pkl", "wb"
    ) as file:
        pickle.dump(data, file)

    # --- Visualize the results --- #
    if vizualize_sol_flag:
        import bioviz

        b = bioviz.Viz(model_path=biorbd_model_path)
        b.load_movement(q_sol)
        b.exec()


if __name__ == "__main__":
    main()
