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
    my_additional_factor=1,
) -> DynamicsEvaluation:


    # @Ipuch: is this really implicit dynamics ?
    # function G = G_Trapezoidal(X_i, X_i_plus, dX_i, dX_i_plus, dt)
    # G =  X_i_plus - (X_i + (dX_i + dX_i_plus)/2*dt);

    # f_G_Trapezoidal = Function('f_G_Trapezoidal', {X_MX, X_plus_MX, dX_MX, dX_plus_MX},
    #                            {G_Trapezoidal(X_MX, X_plus_MX, dX_MX, dX_plus_MX, auxdata.dt)})

    # X_i = X(:, i);
    # X_i_plus = X(:, i + 1);
    # e_ff_i = e_ff(:, i);
    # e_ff_i_plus = e_ff(:, i + 1);
    #
    # dX_i = functions.f_forwardMusculoskeletalDynamics(X_i, e_ff_i, 0, 0, 0 * wM, 0 * wPq, 0 * wPqdot);
    # dX_i_plus = functions.f_forwardMusculoskeletalDynamics(X_i_plus, e_ff_i_plus, 0, 0, 0 * wM, 0 * wPq, 0 * wPqdot);
    # opti.subject_to(functions.f_G_Trapezoidal(X_i, X_i_plus, dX_i, dX_i_plus) * 1e3 == 0);

    #
    # if (
    #         rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS
    #         or rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
    # ):
    #     # contacts forces are directly handled with this constraint
    #     ocp.implicit_constraints.add(
    #         ImplicitConstraintFcn.QDDOT_EQUALS_FORWARD_DYNAMICS,
    #         node=Node.ALL_SHOOTING,
    #         constraint_type=ConstraintType.IMPLICIT,
    #         with_contact=with_contact,
    #         phase=nlp.phase_idx,
    #         with_passive_torque=with_passive_torque,
    #         with_ligament=with_ligament,
    #     ) # G =  X_i_plus - (X_i + (dX_i + dX_i_plus)/2*dt);

    # dX = forwardMusculoskeletalDynamics_motorNoise(X, u, T_EXT, wM, auxdata)
    # a = X(1:6);
    # q = X(7:8);
    # qdot = X(9:10);
    #
    # [Fa, Fp, ~, ~, ~, ~, ~, ~, ~] = getMuscleForce(q, qdot, auxdata);
    #
    # Fm = a. * Fa + Fp;
    # T = TorqueForceRelation(Fm, q, auxdata) + wM;
    #
    # F_forceField = auxdata.forceField * (auxdata.l1 * cos(q(1,:)) + auxdata.l2 * cos(q(1,:)+q(2,:)));
    # T_forceField = -F_forceField * [auxdata.l2 * sin(q(1,:)+q(2,:))+auxdata.l1 * sin(q(1,:));auxdata.l2 * sin(q(1,:)+q(
    #     2,:))];
    #
    #
    # ddtheta = armForwardDynamics(T, q(2), qdot, T_EXT + T_forceField, auxdata);
    #
    # dX = [(u - a). / auxdata.tau;
    # qdot;
    # ddtheta];
    return

def stochastic_forward_dynamics(
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    nlp: NonLinearProgram,
    my_additional_factor=1,
) -> DynamicsEvaluation:

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


def configure_optimal_feedback_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram):

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
    return cas.trace(P_matrix)[key]

def expected_feedback_effort(controller: PenaltyController, final_time: float) -> cas.MX:

    # Constants TODO: remove fom here
    # TODO: How do we choose?
    wM_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024
    dt = final_time / controller.ns
    wM_magnitude = (wM_std * np.ones((2, 1))) ** 2 / dt
    wPq_magnitude = (wPq_std * np.ones((2, 1))) ** 2 / dt
    wPqdot_magnitude = (wPqdot_std * np.ones((2, 1))) ** 2 / dt
    sensoryNoise = np.array([wM_magnitude, wPq_magnitude, wPqdot_magnitude]) @ np.eye(
        controller.states["q"].cx.shape[0] * 3)

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
    hand_pos = controller.model.marker(controller.states["q"].cx_start)[2].to_mx()
    hand_vel = controller.model.marker_velocities(controller.states["q"].cx_start, controller.states["dot"].cx_start)[2].to_mx()
    e_fb = K_matrix @ ((cas.vertcat(hand_pos, hand_vel) - ee_ref) + np.diag(sensoryNoise[2:, 2:]))
    jac_efb_x = cas.jacobian(e_fb, X)
    expectedEffort_fb_MX = cas.trace(jac_efb_x @ P_matrix @ jac_efb_x.T) + cas.trace(K_matrix @ sensoryNoise @ K_matrix.T)
    f_expectedEffort_fb = cas.Function('f_expectedEffort_fb', [controller.states.cx_start, controller.stochastic_variables.cx_start], [expectedEffort_fb_MX])(controller.states.cx_start, controller.stochastic_variables.cx_start)
    return f_expectedEffort_fb

def prepare_ofcp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
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
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscle", wight=1e3/2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="muscle", wight=1e3/2)
    objective_functions.add(minimize_uncertainty,  custom_type=ObjectiveFcn.Lagrange, key="muscle", wight=1e3/2)
    objective_functions.add(expected_feedback_effort, custom_type=ObjectiveFcn.Lagrange, wight=1e3)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(configure_optimal_feedback_problem, dynamic_function=optimal_feedback_forward_dynamics, final_time=final_time)

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
    n_tau = bio_model.nb_tau
    # u_bounds = Bounds([-100] * n_tau + [0] * n_muscles, [100] * n_tau + [1] * n_muscles)
    u_bounds = Bounds([0.01] * n_muscles, [1] * n_muscles)

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

    n_stochastic = n_muscles*n_q + n_q+n_qdot + n_states*n_states + n_states*n_states  # K(6x4) + ee_ref(4x1) + M(10x10) + P(10x10)
    s_bounds = Bounds([-10] * n_stochastic, [10] * n_stochastic)
    stochastic_init = np.zeros((n_stochastic, 1))
    stochastic_init[:n_muscles*n_q, 0] = 0.01  # K
    stochastic_init[n_muscles*n_q:n_muscles*n_q + n_q+n_qdot, 0] = 0.01  # ee_ref
    stochastic_init[n_muscles*n_q + n_q+n_qdot:n_muscles*n_q + n_q+n_qdot + n_states*n_states, 0] = 0.01  # M
    mat_p_init = np.eye(10) * np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-7, 1e-7])
    stochastic_init[n_muscles*n_q + n_q+n_qdot:n_muscles*n_q + n_q+n_qdot + n_states*n_states:, 0] = mat_p_init.flatten()  # P

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
        ode_solver=ode_solver,
        n_threads=n_threads,
        assume_phase_dynamics=False,  # TODO: see if it can be done with assume_phase_dynamics=True
        problem_type=OcpType.OFCP, # StochasticOPtim... (comme mhe)
    )


def prepare_socp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
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
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscle", wight=1)
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

    K = opti.variable(6 * 4, N + 1);
    opti.set_initial(K, 0.01);
    EE_ref = opti.variable(4, N + 1);
    opti.set_initial(EE_ref, 0.01);
    M = opti.variable(nStates, nStates * N);
    opti.set_initial(M, 0.01);
    Pmat_init = [1e-6;1e-6;1e-6;1e-6;1e-6;1e-6;1e-4;1e-4;1e-7;1e-7;].*eye(10);


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
        ode_solver=ode_solver,
        n_threads=n_threads,
        assume_phase_dynamics=False,  # TODO: see if it can be done with assume_phase_dynamics=True
    )


def main():

    import bioviz
    b = bioviz.Viz("models/LeuvenArmModel.bioMod")
    b.exec()

    model = biorbd.Model
    R = biorbd.Rotation.fromEulerAngles(np.array([-np.pi/2, np.pi, np.pi]), 'zxx').to_array()
    R = biorbd.Rotation.fromEulerAngles(np.array([np.pi, np.pi]), 'xz').to_array()


    # --- Prepare the ocp --- #
    # TODO change the model to their model
    dt = 0.1  # 0.01
    final_time = 0.8
    n_shooting = int(final_time/dt)

    # TODO: devrait-il y avoir ns ou ns+1 P ?
    ocp = prepare_ofcp(biorbd_model_path="models/LeuvenArmModel.bioMod", final_time=final_time, n_shooting=n_shooting)

    ocp = prepare_socp(biorbd_model_path="models/LeuvenArmModel.bioMod", final_time=final_time, n_shooting=n_shooting)

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
