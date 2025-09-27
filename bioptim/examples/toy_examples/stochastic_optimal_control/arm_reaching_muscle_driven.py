"""
This example replicates the results from "An approximate stochastic optimal control framework to simulate nonlinear
neuro-musculoskeletal models in the presence of noise"(https://doi.org/10.1371/journal.pcbi.1009338).
The task is to unfold the arm to reach a target further from the trunk.
Noise is added on the motor execution (motor_noise) and on the feedback (wEE=wP and wEE_dot=wPdot).
The expected joint angles (x_mean) are optimized like in a deterministic OCP, but the covariance matrix is minimized to
reduce uncertainty. This covariance matrix is computed from the expected states.

Note: In the original paper, the continuity constraint was weighted (*1e3), but as we do not encourage users to weight
constraint, this feature is not implemented in bioptim (if you really want this feature, please notify the developers
by opening an issue on GitHub). However, the equivalence of our implementation has been tested.

WARNING: These examples are not maintained anymore, please use SocpType.COLLOCATION for a safer, faster, better alternative.
"""

import pickle

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

from bioptim import (
    OptimalControlProgram,
    StochasticOptimalControlProgram,
    StochasticBioModel,
    PhaseDynamics,
    InitialGuessList,
    ObjectiveFcn,
    Solver,
    ObjectiveList,
    NonLinearProgram,
    DynamicsEvaluation,
    DynamicsFunctions,
    ConfigureProblem,
    ConfigureVariables,
    DynamicsOptionsList,
    BoundsList,
    InterpolationType,
    SocpType,
    PenaltyController,
    Node,
    ConstraintList,
    ConstraintFcn,
    MultinodeConstraintList,
    MultinodeObjectiveList,
    ControlType,
    ContactType,
    BiMapping,
)
from bioptim.examples.utils import ExampleUtils
from bioptim.examples.toy_examples.stochastic_optimal_control.arm_reaching_torque_driven_implicit import ExampleType
from bioptim.examples.toy_examples.stochastic_optimal_control.models.leuven_arm_model import LeuvenArmModel


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
    hand_pos_velo = nlp.model.end_effector_pos_velo(q, qdot)
    return hand_pos_velo


def stochastic_forward_dynamics(
    time: cas.MX | cas.SX,
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    algebraic_states: cas.MX | cas.SX,
    numerical_timeseries: cas.MX | cas.SX,
    nlp: NonLinearProgram,
    force_field_magnitude,
    with_noise,
) -> DynamicsEvaluation:
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    mus_activations = DynamicsFunctions.get(nlp.states["muscles"], states)
    mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)

    motor_noise = 0
    sensory_noise = 0
    if with_noise:
        motor_noise = DynamicsFunctions.get(nlp.parameters["motor_noise"], parameters)
        sensory_noise = DynamicsFunctions.get(nlp.parameters["sensory_noise"], parameters)

    mus_excitations_fb = mus_excitations
    noise_torque = np.zeros(nlp.model.motor_noise_magnitude.shape)
    if with_noise:
        ref = DynamicsFunctions.get(nlp.controls["ref"], controls)
        k = DynamicsFunctions.get(nlp.controls["k"], controls)
        k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)

        hand_pos_velo = nlp.model.sensory_reference(
            time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp
        )

        mus_excitations_fb += nlp.model.get_excitation_with_feedback(k_matrix, hand_pos_velo, ref, sensory_noise)
        noise_torque = motor_noise

    muscles_tau = nlp.model.get_muscle_torque(q, qdot, mus_activations)

    tau_force_field = nlp.model.force_field(q, force_field_magnitude)

    torques_computed = muscles_tau + noise_torque + tau_force_field
    dq_computed = qdot
    dactivations_computed = (mus_excitations_fb - mus_activations) / nlp.model.tau_coef

    a1 = nlp.model.I1 + nlp.model.I2 + nlp.model.m2 * nlp.model.l1**2
    a2 = nlp.model.m2 * nlp.model.l1 * nlp.model.lc2
    a3 = nlp.model.I2

    theta_elbow = q[1]
    dtheta_shoulder = qdot[0]
    dtheta_elbow = qdot[1]

    cx = type(theta_elbow)
    mass_matrix = cx(2, 2)
    mass_matrix[0, 0] = a1 + 2 * a2 * cas.cos(theta_elbow)
    mass_matrix[0, 1] = a3 + a2 * cas.cos(theta_elbow)
    mass_matrix[1, 0] = a3 + a2 * cas.cos(theta_elbow)
    mass_matrix[1, 1] = a3

    nleffects = cx(2, 1)
    nleffects[0] = a2 * cas.sin(theta_elbow) * (-dtheta_elbow * (2 * dtheta_shoulder + dtheta_elbow))
    nleffects[1] = a2 * cas.sin(theta_elbow) * dtheta_shoulder**2

    friction = nlp.model.friction_coefficients

    dqdot_computed = cas.inv(mass_matrix) @ (torques_computed - nleffects - friction @ qdot)

    return DynamicsEvaluation(dxdt=cas.vertcat(dq_computed, dqdot_computed, dactivations_computed), defects=None)


def configure_stochastic_optimal_control_problem(
    ocp: OptimalControlProgram,
    nlp: NonLinearProgram,
    numerical_data_timeseries=None,
    contact_types: list[ContactType] | tuple[ContactType] = (),
):
    n_noised_states = 10
    n_references = 4
    n_noised_controls = 6

    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_muscles(
        ocp, nlp, as_states=True, as_controls=True
    )  # Muscles activations as states + muscles excitations as controls

    # Algebraic variables
    ConfigureProblem.configure_stochastic_k(ocp, nlp, n_noised_controls=n_noised_controls, n_references=n_references)
    ConfigureProblem.configure_stochastic_ref(ocp, nlp, n_references=n_references)
    ConfigureProblem.configure_stochastic_m(ocp, nlp, n_noised_states=n_noised_states)
    mat_cov_init = cas.DM_eye(10) * np.array([1e-4, 1e-4, 1e-7, 1e-7, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])

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
        initial_matrix=mat_cov_init,
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
            time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp, with_noise=True
        ),
    )


def minimize_uncertainty(controllers: list[PenaltyController], key: str) -> cas.MX:
    """
    Minimize the uncertainty (covariance matrix) of the states.
    """
    dt = controllers[0].dt.cx
    out = 0
    for i, ctrl in enumerate(controllers):
        cov_matrix = StochasticBioModel.reshape_to_matrix(ctrl.integrated_values["cov"].cx, ctrl.model.matrix_shape_cov)
        p_partial = cov_matrix[ctrl.states[key].index, ctrl.states[key].index]
        out += cas.trace(p_partial) * dt

    return out


def get_cov_mat(nlp, node_index):
    dt = nlp.dt

    nlp.states.node_index = node_index - 1
    nlp.controls.node_index = node_index - 1
    nlp.algebraic_states.node_index = node_index - 1
    nlp.integrated_values.node_index = node_index - 1

    m_matrix = StochasticBioModel.reshape_to_matrix(nlp.algebraic_states["m"].cx, nlp.model.matrix_shape_m)

    CX_eye = cas.SX_eye if nlp.cx == cas.SX else cas.MX_eye
    sensory_noise = nlp.parameters["sensory_noise"].cx
    motor_noise = nlp.parameters["motor_noise"].cx
    sigma_w = cas.vertcat(sensory_noise, motor_noise) * CX_eye(6)
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
        force_field_magnitude=nlp.model.force_field_magnitude,
        with_noise=True,
    )

    ddx_dwm = cas.jacobian(dx.dxdt, cas.vertcat(sensory_noise, motor_noise))
    dg_dw = -ddx_dwm * dt
    ddx_dx = cas.jacobian(dx.dxdt, nlp.states.cx)
    dg_dx = -(ddx_dx * dt / 2 + CX_eye(ddx_dx.shape[0]))

    p_next = m_matrix @ (dg_dx @ cov_matrix @ dg_dx.T + dg_dw @ sigma_w @ dg_dw.T) @ m_matrix.T

    parameters = nlp.parameters.cx
    parameters[nlp.parameters["sensory_noise"].index] = nlp.model.sensory_noise_magnitude
    parameters[nlp.parameters["motor_noise"].index] = nlp.model.motor_noise_magnitude

    func_eval = cas.Function(
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
    )(
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


def reach_target_consistantly(controllers: list[PenaltyController]) -> cas.MX:
    """
    Constraint the hand to reach the target consistently.
    This is a multi-node constraint because the covariance matrix depends on all the precedent nodes, but it only
    applies at the END node.
    """

    q_sym = cas.MX.sym("q_sym", controllers[-1].states["q"].cx.shape[0])
    qdot_sym = cas.MX.sym("qdot_sym", controllers[-1].states["qdot"].cx.shape[0])
    cov_sym = cas.MX.sym("cov", controllers[-1].integrated_values.cx.shape[0])
    cov_matrix = StochasticBioModel.reshape_to_matrix(cov_sym, controllers[-1].model.matrix_shape_cov)

    hand_pos = controllers[0].model.end_effector_position(q_sym)
    hand_vel = controllers[0].model.end_effector_velocity(q_sym, qdot_sym)

    jac_marker_q = cas.jacobian(hand_pos, q_sym)
    jac_marker_qdot = cas.jacobian(hand_vel, cas.vertcat(q_sym, qdot_sym))

    cov_matrix_q = cov_matrix[:2, :2]
    cov_matrix_qdot = cov_matrix[:4, :4]

    pos_constraint = jac_marker_q @ cov_matrix_q @ jac_marker_q.T
    vel_constraint = jac_marker_qdot @ cov_matrix_qdot @ jac_marker_qdot.T

    out = cas.vertcat(pos_constraint[0, 0], pos_constraint[1, 1], vel_constraint[0, 0], vel_constraint[1, 1])

    fun = cas.Function("reach_target_consistantly", [q_sym, qdot_sym, cov_sym], [out])
    val = fun(
        controllers[-1].states["q"].cx,
        controllers[-1].states["qdot"].cx,
        controllers[-1].integrated_values.cx,
    )

    return val


def expected_feedback_effort(controllers: list[PenaltyController], sensory_noise_magnitude: cas.DM) -> cas.MX:
    """
    This function computes the expected effort due to the motor command and feedback gains for a given sensory noise
    magnitude.
    It is computed as Jacobian(effort, states) @ cov @ Jacobian(effort, states) +
                        Jacobian(effort, motor_noise) @ sigma_w @ Jacobian(effort, motor_noise)

    Parameters
    ----------
    controllers : list[PenaltyController]
        List of controllers to be used to compute the expected effort.
    sensory_noise_magnitude : cas.DM
        Magnitude of the sensory noise.
    """
    dt = controllers[0].dt.cx
    CX_eye = cas.MX_eye if controllers[0].cx == cas.MX else cas.SX_eye
    sensory_noise_matrix = sensory_noise_magnitude * CX_eye(4)

    # create the casadi function to be evaluated
    # Get the symbolic variables
    ref = controllers[0].controls["ref"].cx
    cov_sym = controllers[0].cx.sym("cov", controllers[0].integrated_values.cx.shape[0])
    cov_matrix = StochasticBioModel.reshape_to_matrix(cov_sym, controllers[0].model.matrix_shape_cov)

    k = controllers[0].controls["k"].cx
    k_matrix = StochasticBioModel.reshape_to_matrix(k, controllers[0].model.matrix_shape_k)

    # Compute the expected effort
    hand_pos_velo = controllers[0].model.sensory_reference(
        controllers[0].time.cx,
        controllers[0].states.cx,
        controllers[0].controls.cx,
        controllers[0].parameters.cx,
        controllers[0].algebraic_states.cx,
        controllers[0].numerical_timeseries.cx,
        controllers[0].get_nlp,
    )
    trace_k_sensor_k = cas.trace(k_matrix @ sensory_noise_matrix @ k_matrix.T)
    e_fb = k_matrix @ ((hand_pos_velo - ref) + sensory_noise_magnitude)
    jac_e_fb_x = cas.jacobian(e_fb, controllers[0].states.cx)
    trace_jac_p_jack = cas.trace(jac_e_fb_x @ cov_matrix @ jac_e_fb_x.T)
    expectedEffort_fb_mx = trace_jac_p_jack + trace_k_sensor_k
    func = cas.Function(
        "f_expectedEffort_fb",
        [controllers[0].states.cx, controllers[0].controls.cx, cov_sym],
        [expectedEffort_fb_mx],
    )

    f_expectedEffort_fb = 0
    for i, controller in enumerate(controllers):
        P_vector = controller.integrated_values.cx
        out = func(controller.states.cx, controller.controls.cx, P_vector)
        f_expectedEffort_fb += out * dt

    return f_expectedEffort_fb


def zero_acceleration(controller: PenaltyController, force_field_magnitude: float) -> cas.MX:
    """
    No acceleration of the joints at the beginning and end of the movement.
    """
    dx = stochastic_forward_dynamics(
        controller.time.cx,
        controller.states.cx,
        controller.controls.cx,
        controller.parameters.cx,
        controller.algebraic_states.cx,
        controller.numerical_timeseries.cx,
        controller.get_nlp,
        force_field_magnitude=force_field_magnitude,
        with_noise=False,
    )
    return dx.dxdt[2:4]


def track_final_marker(controller: PenaltyController) -> cas.MX:
    """
    Track the hand position.
    """
    q = controller.states["q"].cx
    ee_pos = controller.model.end_effector_position(q)
    return ee_pos


def prepare_socp(
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
    example_type:
        The type of problem to solve (ExampleType.CIRCLE or ExampleType.BAR)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = LeuvenArmModel(
        sensory_noise_magnitude=sensory_noise_magnitude,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_reference=sensory_reference,
    )
    bio_model.force_field_magnitude = force_field_magnitude

    shoulder_pos_initial = 0.349065850398866
    shoulder_pos_final = 0.959931088596881
    elbow_pos_initial = 2.245867726451909  # Optimized in Tom's version
    elbow_pos_final = 1.159394851847144  # Optimized in Tom's version

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL, key="muscles", weight=1e3 / 2, quadratic=True
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, node=Node.ALL, key="muscles", weight=1e3 / 2, quadratic=True
    )

    multinode_objectives = MultinodeObjectiveList()
    multinode_objectives.add(
        minimize_uncertainty,
        nodes_phase=[0 for _ in range(n_shooting + 1)],
        nodes=[i for i in range(n_shooting + 1)],
        key="muscles",
        weight=1e3 / 2,
        quadratic=False,
    )
    multinode_objectives.add(
        expected_feedback_effort,
        nodes_phase=[0 for _ in range(n_shooting + 1)],
        nodes=[i for i in range(n_shooting + 1)],
        sensory_noise_magnitude=sensory_noise_magnitude,
        weight=1e3 / 2,
        quadratic=False,
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_STATE, key="q", node=Node.START, target=np.array([shoulder_pos_initial, elbow_pos_initial])
    )
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.START, target=np.array([0, 0]))
    constraints.add(
        zero_acceleration,
        node=Node.START,
        force_field_magnitude=force_field_magnitude,
    )
    constraints.add(track_final_marker, node=Node.END, target=hand_final_position)
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.END, target=np.array([0, 0]))
    constraints.add(
        zero_acceleration,
        node=Node.END,
        force_field_magnitude=force_field_magnitude,
    )  # Not possible sice the control on the last node is NaN
    constraints.add(ConstraintFcn.TRACK_CONTROL, key="muscles", node=Node.ALL, min_bound=0.001, max_bound=1)
    constraints.add(ConstraintFcn.TRACK_STATE, key="muscles", node=Node.ALL, min_bound=0.001, max_bound=1)
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
        reach_target_consistantly,
        nodes_phase=[0 for _ in range(n_shooting + 1)],
        nodes=[i for i in range(n_shooting + 1)],
        min_bound=np.array([-cas.inf, -cas.inf, -cas.inf, -cas.inf]),
        max_bound=np.array([max_bounds_lateral_variation**2, 0.004**2, 0.05**2, 0.05**2]),
    )

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(
        configure_stochastic_optimal_control_problem,
        dynamic_function=lambda time, states, controls, parameters, algebraic_states, dynamincs_constants, nlp, with_noise: stochastic_forward_dynamics(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            dynamincs_constants,
            nlp,
            force_field_magnitude=force_field_magnitude,
            with_noise=with_noise,
        ),
        expand_dynamics=False,
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        numerical_data_timeseries=None,
    )

    n_muscles = 6
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    n_states = n_q + n_qdot + n_muscles

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
        "muscles",
        min_bound=states_min[n_q + n_qdot :, :],
        max_bound=states_max[n_q + n_qdot :, :],
        interpolation=InterpolationType.EACH_FRAME,
    )

    u_bounds = BoundsList()
    u_bounds.add(
        "muscles",
        min_bound=np.ones((n_muscles,)) * -cas.inf,
        max_bound=np.ones((n_muscles,)) * cas.inf,
        interpolation=InterpolationType.CONSTANT,
    )
    u_bounds.add(
        "k",
        min_bound=np.ones((n_muscles * (n_q + n_qdot),)) * -cas.inf,
        max_bound=np.ones((n_muscles * (n_q + n_qdot),)) * cas.inf,
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
    states_init[n_q + n_qdot :, :] = 0.01

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=states_init[:n_q, :], interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", initial_guess=states_init[n_q : n_q + n_qdot, :], interpolation=InterpolationType.EACH_FRAME)
    x_init.add("muscles", initial_guess=states_init[n_q + n_qdot :, :], interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialGuessList()
    u_init.add("muscles", initial_guess=np.ones((n_muscles,)) * 0.01, interpolation=InterpolationType.CONSTANT)
    u_init.add(
        "k",
        initial_guess=np.ones((n_muscles * (n_q + n_qdot),)) * 0.01,
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

    integrated_value_functions = {"cov": get_cov_mat}

    return StochasticOptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
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
    plot_sol_flag = False
    vizualise_sol_flag = False

    biorbd_model_path = ExampleUtils.folder + "/models/LeuvenArmModel.bioMod"

    hand_initial_position = np.array([0.0, 0.2742])  # Directly from Tom's version
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
    solver.set_maximum_iterations(1000)
    solver.set_hessian_approximation("limited-memory")
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    solver.set_nlp_scaling_method("none")

    example_type = ExampleType.CIRCLE
    force_field_magnitude = 0
    socp = prepare_socp(
        final_time=final_time,
        n_shooting=n_shooting,
        hand_final_position=hand_final_position,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        example_type=example_type,
        force_field_magnitude=force_field_magnitude,
    )

    sol_socp = socp.solve(solver)
    from bioptim import SolutionIntegrator, SolutionMerge

    sol_socp.noisy_integrate(integrator=SolutionIntegrator.SCIPY_RK45, to_merge=SolutionMerge.NODES)

    q_sol = sol_socp.states["q"]
    qdot_sol = sol_socp.states["qdot"]
    activations_sol = sol_socp.states["muscles"]
    excitations_sol = sol_socp.controls["muscles"]
    k_sol = sol_socp.controls["k"]
    ref_sol = sol_socp.controls["ref"]
    m_sol = sol_socp.controls["m"]
    cov_sol_vect = sol_socp.integrated_values["cov"]
    cov_sol = np.zeros((10, 10, n_shooting))
    for i in range(n_shooting):
        for j in range(10):
            for k in range(10):
                cov_sol[j, k, i] = cov_sol_vect[j * 10 + k, i]
    stochastic_variables_sol = np.vstack((k_sol, ref_sol, m_sol))
    data = {
        "q_sol": q_sol,
        "qdot_sol": qdot_sol,
        "activations_sol": activations_sol,
        "excitations_sol": excitations_sol,
        "k_sol": k_sol,
        "ref_sol": ref_sol,
        "m_sol": m_sol,
        "cov_sol": cov_sol,
        "stochastic_variables_sol": stochastic_variables_sol,
    }

    # --- Save the results --- #
    with open(f"leuvenarm_muscle_driven_socp_{example_type}_forcefield{force_field_magnitude}.pkl", "wb") as file:
        pickle.dump(data, file)

    # --- Visualize the solution --- #
    if vizualise_sol_flag:
        import bioviz

        b = bioviz.Viz(model_path=biorbd_model_path)
        b.load_movement(q_sol)
        b.exec()

    # --- Plot the results --- #
    if plot_sol_flag:
        model = LeuvenArmModel()
        q_sym = cas.MX.sym("q_sym", 2, 1)
        qdot_sym = cas.MX.sym("qdot_sym", 2, 1)
        hand_pos_fcn = cas.Function("hand_pos", [q_sym], [model.end_effector_position(q_sym)])
        hand_vel_fcn = cas.Function("hand_vel", [q_sym, qdot_sym], [model.end_effector_velocity(q_sym, qdot_sym)])

        time = socp.nlp[0].time.cx
        dt = socp.nlp[0].dt.cx
        states = socp.nlp[0].states.cx
        controls = socp.nlp[0].controls.cx
        parameters = socp.nlp[0].parameters.cx
        algebraic_states = socp.nlp[0].algebraic_states.cx
        numerical_timeseries = socp.nlp[0].numerical_timeseries.cx
        nlp = socp.nlp[0]
        out = stochastic_forward_dynamics(
            cas.vertcat(time, time + dt),
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
            force_field_magnitude=force_field_magnitude,
            with_noise=True,
        )
        dyn_fun = cas.Function(
            "dyn_fun", [dt, time, states, controls, parameters, algebraic_states, numerical_timeseries], [out.dxdt]
        )

        fig, axs = plt.subplots(3, 2)
        n_simulations = 30
        q_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        qdot_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        mus_activation_simulated = np.zeros((n_simulations, 6, n_shooting + 1))
        hand_pos_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        hand_vel_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        dt_actual = final_time / n_shooting
        for i_simulation in range(n_simulations):
            motor_noise = np.random.normal(0, motor_noise_std, (2, n_shooting + 1))
            wPq = np.random.normal(0, wPq_std, (2, n_shooting + 1))
            wPqdot = np.random.normal(0, wPqdot_std, (2, n_shooting + 1))
            sensory_noise = cas.vertcat(wPq, wPqdot)
            q_simulated[i_simulation, :, 0] = q_sol[:, 0]
            qdot_simulated[i_simulation, :, 0] = qdot_sol[:, 0]
            mus_activation_simulated[i_simulation, :, 0] = activations_sol[:, 0]
            for i_node in range(n_shooting):
                x_prev = cas.vertcat(
                    q_simulated[i_simulation, :, i_node],
                    qdot_simulated[i_simulation, :, i_node],
                    mus_activation_simulated[i_simulation, :, i_node],
                )
                hand_pos_simulated[i_simulation, :, i_node] = np.reshape(hand_pos_fcn(x_prev[:2])[:2], (2,))
                hand_vel_simulated[i_simulation, :, i_node] = np.reshape(
                    hand_vel_fcn(x_prev[:2], x_prev[2:4])[:2], (2,)
                )
                u = excitations_sol[:, i_node]
                s = stochastic_variables_sol[:, i_node]
                k1 = dyn_fun(
                    cas.vertcat(dt_actual * i_node, dt_actual),
                    x_prev,
                    u,
                    [],
                    s,
                    motor_noise[:, i_node],
                    sensory_noise[:, i_node],
                )
                x_next = x_prev + dt * dyn_fun(
                    cas.vertcat(dt_actual * i_node, dt_actual),
                    x_prev + dt / 2 * k1,
                    u,
                    [],
                    s,
                    motor_noise[:, i_node],
                    sensory_noise[:, i_node],
                )
                q_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[:2], (2,))
                qdot_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[2:4], (2,))
                mus_activation_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[4:], (6,))
            hand_pos_simulated[i_simulation, :, i_node + 1] = np.reshape(hand_pos_fcn(x_next[:2])[:2], (2,))
            hand_vel_simulated[i_simulation, :, i_node + 1] = np.reshape(
                hand_vel_fcn(x_next[:2], x_next[2:4])[:2], (2,)
            )
            axs[0, 0].plot(
                hand_pos_simulated[i_simulation, 0, :], hand_pos_simulated[i_simulation, 1, :], color="tab:red"
            )
            axs[1, 0].plot(np.linspace(0, final_time, n_shooting + 1), q_simulated[i_simulation, 0, :], color="k")
            axs[2, 0].plot(np.linspace(0, final_time, n_shooting + 1), q_simulated[i_simulation, 1, :], color="k")
            axs[0, 1].plot(
                hand_vel_simulated[i_simulation, 0, :], hand_vel_simulated[i_simulation, 1, :], color="tab:red"
            )
            axs[1, 1].plot(np.linspace(0, final_time, n_shooting + 1), qdot_simulated[i_simulation, 0, :], color="k")
            axs[2, 1].plot(np.linspace(0, final_time, n_shooting + 1), qdot_simulated[i_simulation, 1, :], color="k")
        hand_pos_without_noise = np.zeros((2, n_shooting + 1))
        for i_node in range(n_shooting + 1):
            hand_pos_without_noise[:, i_node] = np.reshape(hand_pos_fcn(q_sol[:, i_node])[:2], (2,))
        axs[0, 0].plot(hand_pos_without_noise[0, :], hand_pos_without_noise[1, :], color="k")
        axs[0, 0].plot(hand_initial_position[0], hand_initial_position[1], color="tab:green", marker="o")
        axs[0, 0].plot(hand_final_position[0], hand_final_position[1], color="tab:red", marker="o")
        axs[0, 0].set_xlabel("X [m]")
        axs[0, 0].set_ylabel("Y [m]")
        axs[0, 0].set_title("Hand position simulated")
        axs[1, 0].set_xlabel("Time [s]")
        axs[1, 0].set_ylabel("Shoulder angle [rad]")
        axs[2, 0].set_xlabel("Time [s]")
        axs[2, 0].set_ylabel("Elbow angle [rad]")
        axs[0, 1].set_xlabel("X velocity [m/s]")
        axs[0, 1].set_ylabel("Y velocity [m/s]")
        axs[0, 1].set_title("Hand velocity simulated")
        axs[1, 1].set_xlabel("Time [s]")
        axs[1, 1].set_ylabel("Shoulder velocity [rad/s]")
        axs[2, 1].set_xlabel("Time [s]")
        axs[2, 1].set_ylabel("Elbow velocity [rad/s]")
        axs[0, 0].axis("equal")
        plt.tight_layout()
        plt.savefig("simulated_results.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    main()
