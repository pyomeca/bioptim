"""
This example replicates the results from "An approximate stochastic optimal control framework to simulate nonlinear
neuro-musculoskeletal models in the presence of noise"(https://doi.org/10.1371/journal.pcbi.1009338).

The task is to unfold the arm to reach a target further from the trunk.
Noise is added on the motor execution (wM) and on the feedback (wEE and wEE_dot).
The expected joint angles (x_mean) are optimized like in a deterministic OCP, but the covariance matrix is minimized to
reduce uncertainty. This covariance matrix is computed from the expected states.

The present code is composed of two steps:
1) An optimal feedback controls problem in solved to find the optimal deterministic hand trajectory (EE_ref) and the
    optimal gains (K).
2) A stochastic optimal control problem is solved to find the optimal expected states (x_mean) and the optimal
    covariance matrix (P) associated.
"""
import platform

import biorbd
import casadi as cas
import numpy as np

from bioptim import (
    OptimalControlProgram,
    Bounds,
    InitialGuess,
    ObjectiveFcn,
    OdeSolver,
    OdeSolverBase,
    Solver,
    BiorbdModel,
    ObjectiveList,
    NonLinearProgram,
    DynamicsEvaluation,
    DynamicsFunctions,
    ConfigureProblem,
    DynamicsList,
    VariableScalingList,
    BoundsList,
    InterpolationType,
    OcpType,
    PenaltyController,
    Node,
)

def optimal_feedback_forward_dynamics(
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    nlp: NonLinearProgram,
    wM,
) -> DynamicsEvaluation:

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    mus_activations = DynamicsFunctions.get(nlp.states["muscles"], states)
    # residual_tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)

    import biorbd_casadi as biorbd  # Pariterre: using controller.model.forward_dynamics gives free variables error ?
    model = biorbd.Model(
        "/home/charbie/Documents/Programmation/BiorbdOptim/bioptim/examples/stochastic_optimal_control/models/arm26.bioMod")  # controller.get_nlp.model.model

    muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations)

    torques_computed = muscles_tau + wM  # + residual_tau
    dq_computed = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    dactivations_computed = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations)

    # #TODO: add force field

    friction = np.array([[0.05, 0.025], [0.025, 0.05]])
    mass_matrix = model.massMatrix(q).to_mx()
    nleffects = model.NonLinearEffect(q, qdot).to_mx()

    dqdot_derivative = cas.inv(mass_matrix) @ (torques_computed - nleffects - friction @ qdot)

    return DynamicsEvaluation(dxdt=cas.vertcat(dq_computed, dqdot_derivative, dactivations_computed), defects=None)

def stochastic_forward_dynamics(
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    nlp: NonLinearProgram,
    my_additional_factor=1,
) -> DynamicsEvaluation:

    # Constants TODO: remove fom here
    wM_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024
    dt = final_time / controller.ns
    wM_magnitude = (wM_std * np.ones((2, 1))) ** 2 / dt
    wPq_magnitude = (wPq_std * np.ones((2, 1))) ** 2 / dt
    wPqdot_magnitude = (wPqdot_std * np.ones((2, 1))) ** 2 / dt
    sensory_noise = np.array([wM_magnitude, wPq_magnitude, wPqdot_magnitude]) @ np.eye(
        controller.states["q"].cx.shape[0] * 3)

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    muscle_excitation = DynamicsFunctions.get(nlp.controls["muscles"], states)
    muscle_activation = DynamicsFunctions.get(nlp.controls["muscles"], controls)

    M = nlp.model.Inertia(q[:, i])
    joint_friction = np.array([[0.05, 0.025], [0.025, 0.05]])

    # ddtheta(:, i) = M\(T(:, i) + T_EXT(:, i) - C - B * qdot(:, i));

    # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot) * my_additional_factor
    ddq = nlp.model.forward_dynamics(q, qdot, tau)

    # the user has to choose if want to return the explicit dynamics dx/dt = f(x,u,p)
    # as the first argument of DynamicsEvaluation or
    # the implicit dynamics f(x,u,p,xdot)=0 as the second argument
    # which may be useful for IRK or COLLOCATION integrators
    return DynamicsEvaluation(dxdt=cas.vertcat(dq, ddq), defects=None)


def configure_optimal_feedback_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram, wM):

    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)
    # ConfigureProblem.configure_tau(ocp, nlp, False, True)  # Residual tau
    # TODO: eux ils n'utilsent pas de torques résiduels, si les muscles sont plannaire, ce devrit être good
    ConfigureProblem.configure_muscles(ocp, nlp, True, True)  # Muscles activation (states) + excitation (control)

    # Stochastic variables
    ConfigureProblem.configure_k(ocp, nlp)
    ConfigureProblem.configure_ee_ref(ocp, nlp)
    ConfigureProblem.configure_m(ocp, nlp)
    ConfigureProblem.configure_cov(ocp, nlp)
    # ?
    # ConfigureProblem.configure_c(ocp, nlp)
    # ConfigureProblem.configure_a(ocp, nlp)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, optimal_feedback_forward_dynamics, wM=wM, expand=False)

def configure_stochastic_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram):

    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)
    # ConfigureProblem.configure_tau(ocp, nlp, False, True)  # Residual tau
    # TODO: eux ils n'utilsent pas de torques résiduels, si les muscles sont plannaire, ce devrit être good
    ConfigureProblem.configure_muscles(ocp, nlp, True, True)  # Muscles activation (states) + excitation (control)

    # Stochastic variables
    ConfigureProblem.configure_k(ocp, nlp)
    ConfigureProblem.configure_c(ocp, nlp)
    ConfigureProblem.configure_a(ocp, nlp)
    ConfigureProblem.configure_cov(ocp, nlp)
    ConfigureProblem.configure_w_motor(ocp, nlp)
    ConfigureProblem.configure_w_position_feedback(ocp, nlp)
    ConfigureProblem.configure_w_velocity_feedback(ocp, nlp)

def minimize_uncertainty(controller: PenaltyController, key: str) -> cas.MX:
    """
    Minimize the uncertainty (covariance matrix) of the states.
    """
    P_matrix = controller.restore_matrix_form_from_vector(controller.stochastic_variables, controller.states.cx.shape[0], controller.states.cx.shape[0], Node.START, "cov")
    P_partial = P_matrix[controller.states[key].index, controller.states[key].index]
    return cas.trace(P_partial)

def expected_feedback_effort(controller: PenaltyController, final_time: float) -> cas.MX:

    # Constants TODO: remove fom here
    # TODO: How do we choose?
    wM_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024
    dt = final_time / controller.ns
    wM_magnitude = cas.DM(np.array([wM_std ** 2 / dt, wM_std ** 2 / dt]))
    wPq_magnitude = cas.DM(np.array([wPq_std ** 2 / dt, wPq_std ** 2 / dt]))
    wPqdot_magnitude = cas.DM(np.array([wPqdot_std ** 2 / dt, wPqdot_std ** 2 / dt]))
    sensory_noise = cas.vertcat(wM_magnitude, wPq_magnitude, wPqdot_magnitude).T @ np.eye(
        controller.states["q"].cx.shape[0] * 3)
    sensory_noise_matrix = sensory_noise[2:].T * cas.MX_eye(4)

    # Get the symbolic variables
    X = controller.states.cx_start
    ee_ref = controller.stochastic_variables["ee_ref"].cx_start
    P_matrix = controller.restore_matrix_form_from_vector(controller.stochastic_variables,
                                                          controller.states.cx.shape[0],
                                                          controller.states.cx.shape[0],
                                                          Node.START,
                                                          "cov")
    K_matrix = controller.restore_matrix_form_from_vector(controller.stochastic_variables,
                                                          controller.states["muscles"].cx.shape[0],
                                                          controller.states["q"].cx.shape[0] + controller.states["qdot"].cx.shape[0],
                                                          Node.START,
                                                          "k")


    # Compute the expected effort
    hand_pos = controller.model.marker(controller.states["q"].cx_start, 2)[:2]
    hand_vel = controller.model.marker_velocities(controller.states["q"].cx_start, controller.states["qdot"].cx_start, 2)[:2]
    trace_k_sensor_k = cas.trace(K_matrix @ sensory_noise_matrix @ K_matrix.T)
    ee = cas.vertcat(hand_pos, hand_vel)
    e_fb = K_matrix @ ((ee - ee_ref) + sensory_noise[2:].T)
    jac_e_fb_x = cas.jacobian(e_fb, controller.states.cx_start)
    trace_jac_p_jack = cas.trace(jac_e_fb_x @ P_matrix @ jac_e_fb_x.T)
    expectedEffort_fb_mx = trace_k_sensor_k + trace_jac_p_jack
    f_expectedEffort_fb = cas.Function('f_expectedEffort_fb', [controller.states.cx_start, controller.stochastic_variables.cx_start], [expectedEffort_fb_mx])(controller.states.cx_start, controller.stochastic_variables.cx_start)
    return f_expectedEffort_fb

def prepare_ofcp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    n_threads: int = 1,
    wM_std: float = 0.05,
) -> OptimalControlProgram:
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
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1e3/2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="muscles", weight=1e3/2)
    objective_functions.add(minimize_uncertainty,  custom_type=ObjectiveFcn.Lagrange, key="muscles", weight=1e3/2)
    objective_functions.add(expected_feedback_effort, custom_type=ObjectiveFcn.Lagrange, weight=1e3, final_time=final_time)

    # Dynamics
    dynamics = DynamicsList()
    dt = final_time / n_shooting
    wM_magnitude = (wM_std * np.ones((2, 1))) ** 2 / dt
    dynamics.add(configure_optimal_feedback_problem, dynamic_function=optimal_feedback_forward_dynamics, wM=wM_magnitude, expand=False)

    # Path constraint
    shoulder_pos_init = 0.3491
    shoulder_pos_final = 0.9599
    elbow_pos_init = 2.2459  # Optimized in Tom's version
    elbow_pos_final = 1.1594  # Optimized in Tom's version

    n_states = bio_model.nb_q + bio_model.nb_qdot + bio_model.nb_muscles

    q_qdot_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    states_min = np.zeros((n_states, 3))
    states_max = np.zeros((n_states, 3))
    states_min[:bio_model.nb_q + bio_model.nb_qdot, :] = q_qdot_bounds.min
    states_max[:bio_model.nb_q + bio_model.nb_qdot, :] = q_qdot_bounds.max
    states_min[bio_model.nb_q + bio_model.nb_qdot:, :] = 0.01  # activations
    states_min[bio_model.nb_q + bio_model.nb_qdot:, :] = 1  # activations
    states_min[0:2, 0] = [shoulder_pos_init, elbow_pos_init]  # Initial position
    states_max[0:2, 0] = [shoulder_pos_init, elbow_pos_init]  # Initial position
    states_min[0:2, 1:3] = [0, 0]
    states_max[0:2, 1:3] = [np.pi, np.pi]
    # Final position is a stochastic constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=Bounds(states_min, states_max))

    n_muscles = bio_model.nb_muscles
    # n_tau = bio_model.nb_tau
    # u_bounds = Bounds([-100] * n_tau + [0] * n_muscles, [100] * n_tau + [1] * n_muscles)
    u_bounds = Bounds([0.01] * n_muscles, [1] * n_muscles)

    # Initial guesses
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot

    states_init = np.zeros((n_states, 2))
    states_init[0, :] = [shoulder_pos_init, shoulder_pos_final]
    states_init[1, :] = [elbow_pos_init, elbow_pos_final]
    states_init[n_q + n_qdot:, :] = 0.01
    x_init = InitialGuess(states_init, interpolation=InterpolationType.LINEAR)

    u_init = InitialGuess([0.01] * n_muscles)

    # TODO: This should probably be done automatically, not defined by the user

    n_stochastic = n_muscles*(n_q + n_qdot) + n_q+n_qdot + n_states*n_states + n_states*n_states  # K(6x4) + ee_ref(4x1) + M(10x10) + P(10x10)
    # 216 ou 24
    s_bounds = Bounds([-10] * n_stochastic, [10] * n_stochastic)
    stochastic_init = np.zeros((n_stochastic, 1))
    curent_index = 0
    stochastic_init[:n_muscles*(n_q + n_qdot), 0] = 0.01  # K
    curent_index += n_muscles*(n_q + n_qdot)
    stochastic_init[curent_index : curent_index + n_q+n_qdot, 0] = 0.01  # ee_ref
    curent_index += n_q+n_qdot
    stochastic_init[curent_index : curent_index + n_states*n_states, 0] = 0.01  # M
    curent_index += n_states*n_states
    mat_p_init = np.eye(10) * np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-7, 1e-7])
    stochastic_init[curent_index:, 0] = mat_p_init.flatten()  # P

    s_init = InitialGuess(stochastic_init)

    # Variable scaling
    x_scaling = VariableScalingList()
    x_scaling.add("q", scaling=[1, 1])
    x_scaling.add("qdot", scaling=[1, 1])

    u_scaling = VariableScalingList()
    u_scaling.add("tau", scaling=[1, 1])
    u_scaling.add("muscles", scaling=[1, 1])

    # TODO: we should probably change the name stochastic_variables -> helper_variables ?
    # TODO: se mettre en implicit collocations

    return OptimalControlProgram(
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
        ode_solver=OdeSolver.RK2(),
        n_threads=n_threads,
        assume_phase_dynamics=False,  # TODO: see if it can be done with assume_phase_dynamics=True
        problem_type=OcpType.OFCP,  # TODO: seems weird for me to do StochasticOPtim... (comme mhe)
    )


def prepare_socp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    n_threads: int = 1,
) -> OptimalControlProgram:
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
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STOCHASTIC_VARIABLE, key="cov", weight=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=forward_dynamics_with_friction)

    # Path constraint
    shoulder_pos_init = 0.3491
    shoulder_pos_final = 0.9599
    elbow_pos_init = 2.2459  # Optimized in Tom's version
    elbow_pos_final = 1.1594  # Optimized in Tom's version

    n_states = bio_model.nb_q + bio_model.nb_qdot + bio_model.nb_muscles

    q_qdot_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    states_min = np.zeros((n_states, 3))
    states_max = np.zeros((n_states, 3))
    states_min[:bio_model.nb_q + bio_model.nb_qdot, :] = q_qdot_bounds.min
    states_max[:bio_model.nb_q + bio_model.nb_qdot, :] = q_qdot_bounds.max
    states_min[bio_model.nb_q + bio_model.nb_qdot:, :] = 0  # activations
    states_min[bio_model.nb_q + bio_model.nb_qdot:, :] = 1  # activations
    states_min[0:2, 0] = [shoulder_pos_init, elbow_pos_init]  # Initial position
    states_max[0:2, 0] = [shoulder_pos_init, elbow_pos_init]  # Initial position
    # Final position is a stochastic constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=Bounds(states_min, states_max))

    n_muscles = bio_model.nb_muscles
    n_tau = bio_model.nb_tau
    # u_bounds = Bounds([-100] * n_tau + [0] * n_muscles, [100] * n_tau + [1] * n_muscles)
    u_bounds = Bounds([0] * n_muscles, [1] * n_muscles)

    # Initial guesses
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot

    states_init = np.zeros((n_states, 2))
    states_init[0, :] = [shoulder_pos_init, shoulder_pos_final]
    states_init[1, :] = [elbow_pos_init, elbow_pos_final]
    states_init[n_q + n_qdot:, :] = 0.01
    x_init = InitialGuess([0] * (n_q + n_qdot), interpolation=InterpolationType.LINEAR)

    u_init = InitialGuess([0.01] * n_muscles)

    # TODO: This should probably be done automatically, not defined by the user
    # not right
    n_stochastic = n_muscles*n_q + n_q + n_q**2 + n_q**2 + n_q + n_q + n_q  # + K(6x4)
    s_bounds = Bounds([-10] * n_stochastic, [10] * n_stochastic)
    s_init = InitialGuess([0] * n_stochastic)  # TODO: to be changed probable really bad

    # Variable scaling
    x_scaling = VariableScalingList()
    x_scaling.add("q", scaling=[1, 1])
    x_scaling.add("qdot", scaling=[1, 1])

    u_scaling = VariableScalingList()
    u_scaling.add("tau", scaling=[1, 1])
    u_scaling.add("muscles", scaling=[1, 1])

    # TODO: we should probably change the name stochastic_variables -> helper_variables ?
    # TODO: add end effector position and velocity correction
    # TODO: se mettre en implicit collocations

    return OptimalControlProgram(
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
        ode_solver=OdeSolver.RK2(),
        n_threads=n_threads,
        assume_phase_dynamics=False,  # TODO: see if it can be done with assume_phase_dynamics=True
    )


def main():

    # import bioviz
    # b = bioviz.Viz("models/LeuvenArmModel.bioMod")
    # b.exec()

    # --- Prepare the ocp --- #
    # TODO change the model to their model
    dt = 0.1  # 0.01
    final_time = 0.8
    n_shooting = int(final_time/dt)

    # TODO: devrait-il y avoir ns ou ns+1 P ?
    ofcp = prepare_ofcp(biorbd_model_path="models/LeuvenArmModel.bioMod", final_time=final_time, n_shooting=n_shooting)
    ofcp.solve(Solver.IPOPT(show_online_optim=False))

    socp = prepare_socp(biorbd_model_path="models/LeuvenArmModel.bioMod", final_time=final_time, n_shooting=n_shooting)

    # Custom plots
    # ocp.add_plot_penalty(CostType.ALL)

    # --- If one is interested in checking the conditioning of the problem, they can uncomment the following line --- #
    # ocp.check_conditioning()

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))  # platform.system() == "Linux"
    # sol.graphs()

    # --- Show the results in a bioviz animation --- #
    sol.animate()


if __name__ == "__main__":
    main()
