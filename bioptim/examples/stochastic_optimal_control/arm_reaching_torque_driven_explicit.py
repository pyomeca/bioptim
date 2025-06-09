"""
This example is adapted from arm_reaching_muscle_driven.py to make it torque driven.
The states dynamics is implicit. which allows to minimize the uncertainty on the acceleration of joints.
The algebraic states dynamics is explicit.
"""

import pickle
from typing import Any

import casadi as cas
import numpy as np

from bioptim import (
    OptimalControlProgram,
    StochasticOptimalControlProgram,
    ObjectiveFcn,
    Solver,
    StochasticBiorbdModel,
    StochasticBioModel,
    ObjectiveList,
    NonLinearProgram,
    DynamicsEvaluation,
    DynamicsFunctions,
    ConfigureProblem,
    ConfigureVariables,
    DynamicsList,
    BoundsList,
    InterpolationType,
    SocpType,
    PenaltyController,
    Node,
    ConstraintList,
    ConstraintFcn,
    MultinodeConstraintList,
    MultinodeObjectiveList,
    InitialGuessList,
    Axis,
    ControlType,
    PhaseDynamics,
    BiMapping,
)
from bioptim.examples.stochastic_optimal_control.arm_reaching_torque_driven_implicit import ExampleType
from bioptim.examples.stochastic_optimal_control.common import (
    dynamics_torque_driven_with_feedbacks,
)


def stochastic_forward_dynamics(
    time: cas.MX | cas.SX,
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    algebraic_states: cas.MX | cas.SX,
    numerical_timeseries: cas.MX | cas.SX,
    nlp: NonLinearProgram,
    with_noise: bool,
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
    algebraic_states: MX.sym
        The algebraic_states variables
    nlp: NonLinearProgram
        The current non-linear program
    with_noise: bool
        If noise should be added (including feedback gains)
    """

    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    qddot = DynamicsFunctions.get(nlp.states["qddot"], states)
    qdddot = DynamicsFunctions.get(nlp.controls["qdddot"], controls)

    dqdot_constraint = dynamics_torque_driven_with_feedbacks(
        time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp, with_noise=with_noise
    )
    defects = cas.vertcat(dqdot_constraint - qddot)

    return DynamicsEvaluation(dxdt=cas.vertcat(qdot, dqdot_constraint, qdddot), defects=defects)


def configure_stochastic_optimal_control_problem(
    ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None, contact_types=()
):
    """
    Configure the stochastic optimal control problem.
    """

    n_noised_states = 6
    n_references = 4
    n_noised_controls = 2

    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qddot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdddot(ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

    # Algebraic states variables
    ConfigureProblem.configure_stochastic_k(ocp, nlp, n_noised_controls=n_noised_controls, n_references=n_references)
    ConfigureProblem.configure_stochastic_ref(ocp, nlp, n_references=n_references)
    ConfigureProblem.configure_stochastic_m(ocp, nlp, n_noised_states=n_noised_states)
    mat_p_init = cas.DM_eye(6) * np.array(
        [1e-4, 1e-4, 1e-7, 1e-7, 1e-6, 1e-6]
    )  # P, the noise on the acceleration should be chosen carefully (here arbitrary)

    # Configure explicit cov as an integrated value
    name_cov = []
    for name_1 in [f"X_{i}" for i in range(n_noised_states)]:
        for name_2 in [f"X_{i}" for i in range(n_noised_states)]:
            name_cov += [name_1 + "_&_" + name_2]
    nlp.variable_mappings["cov"] = BiMapping(list(range(n_noised_states**2)), list(range(n_noised_states**2)))
    ConfigureVariables.configure_integrated_value(
        "cov_explicit",
        name_cov,
        ocp,
        nlp,
        initial_matrix=mat_p_init,
    )

    # Configure dynamics
    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp: nlp.dynamics_type.dynamic_function(
            time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp, with_noise=False
        ),
    )
    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp: nlp.dynamics_type.dynamic_function(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
            with_noise=True,
        ),
    )


def minimize_uncertainty(controllers: list[PenaltyController], key: str) -> cas.MX:
    """
    Minimize the uncertainty (covariance matrix) of the states "key".
    """
    dt = controllers[0].dt.cx
    out: Any = 0
    for i, ctrl in enumerate(controllers):
        cov_matrix = StochasticBioModel.reshape_to_matrix(ctrl.integrated_values["cov"].cx, ctrl.model.matrix_shape_cov)
        p_partial = cov_matrix[ctrl.states[key].index, ctrl.states[key].index]
        out += cas.trace(p_partial) * dt
    return out


def sensory_reference(
    time: cas.MX | cas.SX,
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    algebraic_states: cas.MX | cas.SX,
    numerical_timeseries: cas.MX | cas.SX,
    nlp: NonLinearProgram,
):
    """
    This functions returns the sensory reference for the feedback gains.
    """
    q = states[nlp.states["q"].index]
    qdot = states[nlp.states["qdot"].index]
    hand_pos = nlp.model.marker(2)(q, [])[:2]
    hand_vel = nlp.model.marker_velocity(2)(q, qdot, [])[:2]
    hand_pos_velo = cas.vertcat(hand_pos, hand_vel)
    return hand_pos_velo


def get_cov_mat(nlp, node_index, use_sx):
    """
    Perform a trapezoidal integration to get the covariance matrix at the next node.
    It is computed as:
    P_k+1 = M_k(dg/dx @ P_k @ dg/dx + dg/dw @ sigma_w @ dg/dw) @ M_k

    Parameters
    ----------
    nlp: NonLinearProgram
        The current non-linear program.
    node_index: int
        The node index at hich we want to compute the covariance matrix.
    """

    nlp.states.node_index = 0
    nlp.controls.node_index = 0
    nlp.algebraic_states.node_index = 0
    nlp.integrated_values.node_index = 0

    dt = nlp.dt

    M_matrix = StochasticBioModel.reshape_to_matrix(nlp.algebraic_states["m"].cx, nlp.model.matrix_shape_m)

    CX_eye = cas.SX_eye if use_sx else cas.MX_eye
    sensory_noise = nlp.parameters["sensory_noise"].cx
    motor_noise = nlp.parameters["motor_noise"].cx
    sigma_w = cas.vertcat(sensory_noise, motor_noise) * CX_eye(cas.vertcat(sensory_noise, motor_noise).shape[0])
    cov_sym = nlp.cx.sym("cov", nlp.integrated_values.cx.shape[0])
    cov_matrix = StochasticBioModel.reshape_to_matrix(cov_sym, nlp.model.matrix_shape_cov)

    dx = stochastic_forward_dynamics(
        nlp.time_cx,
        nlp.states.cx,
        nlp.controls.cx,
        nlp.parameters.cx,
        nlp.algebraic_states.cx,
        nlp.numerical_timeseries.cx,
        nlp,
        with_noise=True,
    )

    ddx_dwm = cas.jacobian(dx.dxdt, cas.vertcat(sensory_noise, motor_noise))
    dg_dw = -ddx_dwm * dt
    ddx_dx = cas.jacobian(dx.dxdt, nlp.states.cx)
    dg_dx: Any = -(ddx_dx * dt / 2 + CX_eye(ddx_dx.shape[0]))

    p_next = M_matrix @ (dg_dx @ cov_matrix @ dg_dx.T + dg_dw @ sigma_w @ dg_dw.T) @ M_matrix.T
    func = cas.Function(
        "p_next",
        [
            dt,
            nlp.states.cx,
            nlp.controls.cx,
            nlp.parameters.cx,
            nlp.algebraic_states.cx,
            nlp.numerical_timeseries.cx,
            cov_sym,
        ],
        [p_next],
    )

    nlp.states.node_index = node_index - 1
    nlp.controls.node_index = node_index - 1
    nlp.algebraic_states.node_index = node_index - 1
    nlp.integrated_values.node_index = node_index - 1

    parameters = nlp.parameters.cx
    parameters[nlp.parameters["sensory_noise"].index] = nlp.model.sensory_noise_magnitude
    parameters[nlp.parameters["motor_noise"].index] = nlp.model.motor_noise_magnitude

    func_eval = func(
        nlp.dt,
        nlp.states.cx,
        nlp.controls.cx,
        parameters,
        nlp.algebraic_states.cx,
        nlp.numerical_timeseries.cx,
        nlp.integrated_values["cov"].cx,
    )
    p_vector = StochasticBioModel.reshape_to_vector(func_eval)
    return p_vector


def reach_target_consistently(controllers: list[PenaltyController]) -> cas.MX:
    """
    Constraint the hand to reach the target consistently.
    This is a multi-node constraint because the covariance matrix depends on all the precedent nodes, but it only
    applies at the END node.
    """
    q_sym = cas.MX.sym("q_sym", controllers[-1].states["q"].cx.shape[0])
    qdot_sym = cas.MX.sym("qdot_sym", controllers[-1].states["qdot"].cx.shape[0])
    cov_sym = cas.MX.sym("cov", controllers[-1].integrated_values.cx.shape[0])
    cov_matrix = StochasticBioModel.reshape_to_matrix(cov_sym, controllers[-1].model.matrix_shape_cov)

    hand_pos = controllers[0].model.marker(2)(q_sym, [])[:2]
    hand_vel = controllers[0].model.marker_velocity(2)(q_sym, qdot_sym, [])[:2]

    jac_marker_q = cas.jacobian(hand_pos, q_sym)
    jac_marker_qdot = cas.jacobian(hand_vel, cas.vertcat(q_sym, qdot_sym))

    cov_matrix_q = cov_matrix[:2, :2]
    cov_matrix_qdot = cov_matrix[:4, :4]

    pos_constraint = jac_marker_q @ cov_matrix_q @ jac_marker_q.T
    vel_constraint = jac_marker_qdot @ cov_matrix_qdot @ jac_marker_qdot.T

    out = cas.vertcat(pos_constraint[0, 0], pos_constraint[1, 1], vel_constraint[0, 0], vel_constraint[1, 1])

    fun = cas.Function("reach_target_consistently", [q_sym, qdot_sym, cov_sym], [out])
    val = fun(controllers[-1].states["q"].cx, controllers[-1].states["qdot"].cx, controllers[-1].integrated_values.cx)

    return val


def expected_feedback_effort(controllers: list[PenaltyController]) -> cas.MX:
    """
    This function computes the expected effort due to the motor command and feedback gains for a given sensory noise
    magnitude.
    It is computed as Jacobian(effort, states) @ cov @ Jacobian(effort, states) +
                        Jacobian(efforst, motor_noise) @ sigma_w @ Jacobian(efforst, motor_noise)

    Parameters
    ----------
    controllers : list[PenaltyController]
        List of controllers to be used to compute the expected effort.
    """

    dt = controllers[0].dt.cx
    sensory_noise_matrix = controllers[0].model.sensory_noise_magnitude * cas.MX_eye(4)

    # create the casadi function to be evaluated
    # Get the symbolic variables
    ref = controllers[0].controls["ref"].cx
    cov_sym = cas.MX.sym("cov", controllers[0].integrated_values.cx.shape[0])
    cov_matrix = StochasticBioModel.reshape_to_matrix(cov_sym, controllers[0].model.matrix_shape_cov)

    k = controllers[0].controls["k"].cx
    k_matrix = StochasticBioModel.reshape_to_matrix(k, controllers[0].model.matrix_shape_k)

    # Compute the expected effort
    trace_k_sensor_k = cas.trace(k_matrix @ sensory_noise_matrix @ k_matrix.T)
    estimated_ref = controllers[0].model.sensory_reference(
        controllers[0].time.cx,
        controllers[0].states.cx,
        controllers[0].controls.cx,
        controllers[0].parameters.cx,
        controllers[0].algebraic_states.cx,
        controllers[0].numerical_timeseries.cx,
        controllers[0].get_nlp,
    )
    e_fb = k_matrix @ ((estimated_ref - ref) + controllers[0].model.sensory_noise_magnitude)
    jac_e_fb_x = cas.jacobian(e_fb, controllers[0].states.cx)
    trace_jac_p_jack = cas.trace(jac_e_fb_x @ cov_matrix @ jac_e_fb_x.T)
    expected_effort_fb_mx = trace_jac_p_jack + trace_k_sensor_k
    func = cas.Function(
        "expected_effort_fb_mx",
        [controllers[0].states.cx, controllers[0].controls.cx, cov_sym],
        [expected_effort_fb_mx],
    )

    f_expected_effort_fb: Any = 0
    for i, controller in enumerate(controllers):
        P_vector = controller.integrated_values.cx
        out = func(controller.states.cx, controller.controls.cx, P_vector)
        f_expected_effort_fb += out * dt

    return f_expected_effort_fb


def prepare_socp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    hand_final_position: np.ndarray,
    motor_noise_magnitude: cas.DM,
    sensory_noise_magnitude: cas.DM,
    force_field_magnitude: float = 0,
    example_type=ExampleType.CIRCLE,
    use_sx: bool = False,
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
    force_field_magnitude: float
        The magnitude of the force field
    example_type: ExampleType
        The type of problem to solve (ExampleType.CIRCLE or ExampleType.BAR)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = StochasticBiorbdModel(
        biorbd_model_path,
        n_references=4,
        n_feedbacks=4,
        n_noised_states=6,
        n_noised_controls=2,
        sensory_noise_magnitude=sensory_noise_magnitude,
        motor_noise_magnitude=motor_noise_magnitude,
        friction_coefficients=np.array([[0.05, 0.025], [0.025, 0.05]]),
        sensory_reference=sensory_reference,
    )
    bio_model.force_field_magnitude = force_field_magnitude

    n_tau = bio_model.nb_tau
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    n_states = n_q * 3
    n_controls = n_q * 2

    shoulder_pos_initial = 0.349065850398866
    shoulder_pos_final = 0.959931088596881
    elbow_pos_initial = 2.245867726451909  # Optimized in Tom's version
    elbow_pos_final = 1.159394851847144  # Optimized in Tom's version

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL, key="tau", weight=1e3 / 2, quadratic=True
    )

    multinode_objectives = MultinodeObjectiveList()
    multinode_objectives.add(
        minimize_uncertainty,
        nodes_phase=[0 for _ in range(n_shooting + 1)],
        nodes=[i for i in range(n_shooting + 1)],
        key="qddot",
        weight=1e3 / 2,
        quadratic=False,
    )
    multinode_objectives.add(
        expected_feedback_effort,
        nodes_phase=[0 for _ in range(n_shooting + 1)],
        nodes=[i for i in range(n_shooting + 1)],
        weight=1e3 / 2,
        quadratic=False,
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_STATE, key="q", node=Node.START, target=np.array([shoulder_pos_initial, elbow_pos_initial])
    )
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.START, target=np.array([0, 0]))
    constraints.add(ConstraintFcn.TRACK_STATE, key="qddot", node=Node.START, target=np.array([0, 0]))
    constraints.add(
        ConstraintFcn.TRACK_MARKERS, node=Node.END, target=hand_final_position, marker_index=2, axes=[Axis.X, Axis.Y]
    )
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.END, target=np.array([0, 0]))
    constraints.add(ConstraintFcn.TRACK_STATE, key="qddot", node=Node.END, target=np.array([0, 0]))
    constraints.add(
        ConstraintFcn.TRACK_STATE, key="q", node=Node.ALL, min_bound=0, max_bound=180
    )  # This is a bug, it should be in radians

    if example_type == ExampleType.BAR:
        max_bounds_lateral_variation = cas.inf
    elif example_type == ExampleType.CIRCLE:
        max_bounds_lateral_variation = 0.004
    else:
        raise NotImplementedError("Wrong problem type")

    multinode_constraints = MultinodeConstraintList()
    multinode_constraints.add(
        reach_target_consistently,
        nodes_phase=[0 for _ in range(n_shooting + 1)],
        nodes=[i for i in range(n_shooting + 1)],
        min_bound=np.array([-cas.inf, -cas.inf, -cas.inf, -cas.inf]),
        max_bound=np.array([max_bounds_lateral_variation**2, 0.004**2, 0.05**2, 0.05**2]),
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        configure_stochastic_optimal_control_problem,
        dynamic_function=stochastic_forward_dynamics,
        expand_dynamics=False,
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        numerical_data_timeseries=None,
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
    x_bounds.add(
        "qddot",
        min_bound=states_min[n_q + n_qdot : n_q + n_qdot * 2, :],
        max_bound=states_max[n_q + n_qdot : n_q + n_qdot * 2, :],
        interpolation=InterpolationType.EACH_FRAME,
    )

    controls_min = np.ones((n_controls, 3)) * -cas.inf
    controls_max = np.ones((n_controls, 3)) * cas.inf

    u_bounds = BoundsList()
    u_bounds.add("qdddot", min_bound=controls_min[:n_q, :], max_bound=controls_max[:n_q, :])
    u_bounds.add("tau", min_bound=controls_min[n_q:, :], max_bound=controls_max[n_q:, :])
    u_bounds.add(
        "k",
        min_bound=np.ones((n_tau * (n_q + n_qdot),)) * -cas.inf,
        max_bound=np.ones((n_tau * (n_q + n_qdot),)) * cas.inf,
        interpolation=InterpolationType.CONSTANT,
    )
    u_bounds.add(
        "ref",
        min_bound=np.ones((n_q + n_qdot,)) * -cas.inf,
        max_bound=np.ones((n_q + n_qdot,)) * cas.inf,
        interpolation=InterpolationType.CONSTANT,
    )

    a_bounds = BoundsList()
    a_bounds.add(
        "m",
        min_bound=np.ones((n_states * n_states,)) * -cas.inf,
        max_bound=np.ones((n_states * n_states,)) * cas.inf,
        interpolation=InterpolationType.CONSTANT,
    )

    # Initial guesses
    states_init = np.zeros((n_states, n_shooting + 1))
    states_init[0, :] = np.linspace(shoulder_pos_initial, shoulder_pos_final, n_shooting + 1)
    states_init[1, :] = np.linspace(elbow_pos_initial, elbow_pos_final, n_shooting + 1)
    states_init[n_states:, :] = 0.01

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=states_init[:n_q, :], interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", initial_guess=states_init[n_q : n_q + n_qdot, :], interpolation=InterpolationType.EACH_FRAME)
    x_init.add(
        "qddot",
        initial_guess=states_init[n_q + n_qdot : n_q + n_qdot * 2, :],
        interpolation=InterpolationType.EACH_FRAME,
    )

    controls_init = np.ones((n_controls, n_shooting + 1)) * 0.01

    u_init = InitialGuessList()
    u_init.add("qdddot", initial_guess=controls_init[:n_q, :], interpolation=InterpolationType.EACH_FRAME)
    u_init.add("tau", initial_guess=controls_init[n_q:, :], interpolation=InterpolationType.EACH_FRAME)
    u_init.add(
        "k",
        initial_guess=np.ones((n_tau * (n_q + n_qdot),)) * 0.01,
        interpolation=InterpolationType.CONSTANT,
    )
    u_init.add(
        "ref",
        initial_guess=np.ones((n_q + n_qdot,)) * 0.01,
        interpolation=InterpolationType.CONSTANT,
    )

    a_init = InitialGuessList()
    a_init.add(
        "m",
        initial_guess=np.ones((n_states * n_states,)) * 0.01,
        interpolation=InterpolationType.CONSTANT,
    )

    integrated_value_functions = {"cov": lambda nlp, node_index: get_cov_mat(nlp, node_index, use_sx)}

    return StochasticOptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        a_init=a_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        a_bounds=a_bounds,
        objective_functions=objective_functions,
        multinode_objectives=multinode_objectives,
        constraints=constraints,
        multinode_constraints=multinode_constraints,
        control_type=ControlType.CONSTANT_WITH_LAST_NODE,
        n_threads=1,
        problem_type=SocpType.TRAPEZOIDAL_EXPLICIT(),
        integrated_value_functions=integrated_value_functions,
        use_sx=use_sx,
    )


def main():
    # --- Options --- #
    use_sx = False
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
        force_field_magnitude=force_field_magnitude,
        use_sx=use_sx,
    )

    sol_socp = socp.solve(solver)
    # sol_socp.graphs()

    q_sol = sol_socp.states["q"]
    qdot_sol = sol_socp.states["qdot"]
    qddot_sol = sol_socp.states["qddot"]
    qdddot_sol = sol_socp.controls["qdddot"]
    tau_sol = sol_socp.controls["tau"]
    k_sol = sol_socp.controls["k"]
    ref_sol = sol_socp.controls["ref"]
    m_sol = sol_socp.algebraic_states["m"]
    cov_sol_vect = sol_socp.integrated_values["cov"]
    cov_sol = np.zeros((6, 6, n_shooting))
    for i in range(n_shooting):
        for j in range(6):
            for k in range(6):
                cov_sol[j, k, i] = cov_sol_vect[j * 6 + k, i]
    stochastic_variables_sol = np.vstack((k_sol, ref_sol, m_sol))
    data = {
        "q_sol": q_sol,
        "qdot_sol": qdot_sol,
        "qddot_sol": qddot_sol,
        "qdddot_sol": qdddot_sol,
        "tau_sol": tau_sol,
        "k_sol": k_sol,
        "ref_sol": ref_sol,
        "m_sol": m_sol,
        "cov_sol": cov_sol,
        "stochastic_variables_sol": stochastic_variables_sol,
    }

    # --- Save the results --- #
    with open(f"leuvenarm_torque_driven_socp_{example_type}_forcefield{force_field_magnitude}.pkl", "wb") as file:
        pickle.dump(data, file)

    # --- Visualize the results --- #
    if vizualize_sol_flag:
        import bioviz

        b = bioviz.Viz(model_path=biorbd_model_path)
        b.load_movement(q_sol)
        b.exec()


if __name__ == "__main__":
    main()
