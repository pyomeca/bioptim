import os
import pytest

import numpy as np
from casadi import DM, vertcat
from bioptim import Solver, SocpType, SolutionMerge


@pytest.mark.parametrize("use_sx", [False, True])
def test_arm_reaching_torque_driven_collocations(use_sx: bool):
    from bioptim.examples.stochastic_optimal_control import arm_reaching_torque_driven_collocations as ocp_module

    final_time = 0.4
    n_shooting = 4
    hand_final_position = np.array([9.359873986980460e-12, 0.527332023564034])

    dt = 0.05
    motor_noise_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024
    motor_noise_magnitude = DM(np.array([motor_noise_std**2 / dt, motor_noise_std**2 / dt]))
    wPq_magnitude = DM(np.array([wPq_std**2 / dt, wPq_std**2 / dt]))
    wPqdot_magnitude = DM(np.array([wPqdot_std**2 / dt, wPqdot_std**2 / dt]))
    sensory_noise_magnitude = vertcat(wPq_magnitude, wPqdot_magnitude)

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_socp(
        biorbd_model_path=bioptim_folder + "/models/LeuvenArmModel.bioMod",
        final_time=final_time,
        n_shooting=n_shooting,
        polynomial_degree=3,
        hand_final_position=hand_final_position,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        use_sx=use_sx,
    )

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_nlp_scaling_method("none")

    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 426.84572091057413)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (442, 1))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    algebraic_states = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)

    q, qdot = states["q"], states["qdot"]
    tau = controls["tau"]
    k, ref, m, cov = algebraic_states["k"], algebraic_states["ref"], algebraic_states["m"], algebraic_states["cov"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.9256103, 1.29037205]))
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

    np.testing.assert_almost_equal(tau[:, 0], np.array([1.72235954, -0.90041542]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-1.64870266, 1.08550928]))

    np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412560e-01, 0, 0]))

    np.testing.assert_almost_equal(
        m[:10, 0],
        np.array(
            [
                1.00000000e00,
                6.56332273e-28,
                1.74486171e-28,
                2.01948392e-28,
                -3.89093870e-28,
                1.00000000e00,
                1.56788574e-28,
                -2.17796617e-28,
                5.05560847e-28,
                2.85012070e-27,
            ]
        ),
    )  # suspicious

    # Had to change the values here :(
    np.testing.assert_almost_equal(
        cov[:, -1],  # TODO: This is probably -1 now
        np.array(
            [
                -0.89353026,
                -0.66059229,
                -0.39031482,
                -0.31941486,
                -0.66059229,
                -0.44897437,
                -0.12298423,
                -0.30298653,
                -0.39031482,
                -0.12298423,
                -0.36377371,
                0.05702737,
                -0.31941486,
                -0.30298653,
                0.05702737,
                -0.24391646,
            ]
        ),
    )

    # Test the automatic intialization of the stochastic variables
    socp = ocp_module.prepare_socp(
        biorbd_model_path=bioptim_folder + "/models/LeuvenArmModel.bioMod",
        final_time=final_time,
        n_shooting=n_shooting,
        polynomial_degree=3,
        hand_final_position=hand_final_position,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        q_opt=q,
        qdot_opt=qdot,
        tau_opt=tau,
    )

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_nlp_scaling_method("none")
    solver.set_maximum_iterations(0)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)

    sol_socp = socp.solve(solver)

    # Check some of the results
    states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)
    algebraic_states = sol_socp.decision_algebraic_states(to_merge=SolutionMerge.NODES)
    duration = sol_socp.decision_time()[-1]
    dt = duration / n_shooting

    q_sol, qdot_sol = states["q"], states["qdot"]
    tau_sol = controls["tau"]
    p_sol = vertcat(ocp.nlp[0].model.motor_noise_magnitude, ocp.nlp[0].model.sensory_noise_magnitude)
    k_sol, ref_sol, m_sol, cov_sol = (
        algebraic_states["k"],
        algebraic_states["ref"],
        algebraic_states["m"],
        algebraic_states["cov"],
    )

    polynomial_degree = socp.nlp[0].ode_solver.polynomial_degree

    # Constraint values
    x_opt = vertcat(q_sol, qdot_sol)
    x_sol = np.zeros((x_opt.shape[0], polynomial_degree + 2, socp.n_shooting))
    for i_node in range(socp.n_shooting):
        x_sol[:, :, i_node] = x_opt[:, i_node * (polynomial_degree + 2) : (i_node + 1) * (polynomial_degree + 2)]
    a_sol = vertcat(k_sol, ref_sol, m_sol, cov_sol)

    x_multi_thread = np.zeros((socp.nlp[0].states.shape * (polynomial_degree + 3), socp.nlp[0].ns))
    for i_node in range(socp.nlp[0].ns):
        for i_state in range(socp.nlp[0].states.shape):
            x_multi_thread[i_state, i_node] = x_sol[i_state, 0, i_node]
            for i_coll in range(1, polynomial_degree + 3):
                x_multi_thread[i_coll * socp.nlp[0].states.shape + i_state, i_node] = x_sol[i_state, i_coll - 1, i_node]

    # Initial posture
    shoulder_pos_initial = 0.349065850398866
    elbow_pos_initial = 2.245867726451909
    constraint_value = (
        socp.nlp[0]
        .g[0]
        .function[0](
            duration,
            dt,
            x_multi_thread[:, 0],
            vertcat(tau_sol[:, 0], tau_sol[:, 1]),
            p_sol,
            vertcat(a_sol[:, 0], a_sol[:, 1]),
        )
    )
    np.testing.assert_almost_equal(constraint_value[0], shoulder_pos_initial, decimal=6)
    np.testing.assert_almost_equal(constraint_value[1], elbow_pos_initial, decimal=6)

    # Initial and final velocities
    constraint_value = (
        socp.nlp[0]
        .g[1]
        .function[0](
            duration,
            dt,
            x_multi_thread[:, 0],
            vertcat(tau_sol[:, 0], tau_sol[:, 1]),
            p_sol,
            vertcat(asol[:, 0], asol[:, 1]),
        )
    )
    np.testing.assert_almost_equal(constraint_value[0], 0, decimal=6)
    np.testing.assert_almost_equal(constraint_value[1], 0, decimal=6)
    constraint_value = (
        socp.nlp[0]
        .g[2]
        .function[-1](
            duration,
            dt,
            x_opt[:, -1],
            tau_sol[:, -1],
            p_sol,
            asol[:, -1],
        )
    )
    np.testing.assert_almost_equal(constraint_value[0], 0, decimal=6)
    np.testing.assert_almost_equal(constraint_value[1], 0, decimal=6)

    # Hand final marker position
    constraint_value = (
        socp.nlp[0]
        .g[4]
        .function[-1](
            duration,
            dt,
            x_opt[:, -1],
            tau_sol[:, -1],
            p_sol,
            asol[:, -1],
        )
    )
    np.testing.assert_almost_equal(constraint_value[0], hand_final_position[0], decimal=6)
    np.testing.assert_almost_equal(constraint_value[1], hand_final_position[1], decimal=6)

    # Reference equals mean sensory input
    for i_node in range(socp.n_shooting):
        constraint_value = (
            socp.nlp[0]
            .g[7]
            .function[i_node](
                duration,
                dt,
                x_multi_thread[:, i_node],
                vertcat(tau_sol[:, i_node], tau_sol[:, i_node + 1]),
                p_sol,
                vertcat(asol[:, i_node], asol[:, i_node + 1]),
            )
        )
        np.testing.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)

    # @pariterre how is the new way to use these?
    # Constraint on M --------------------------------------------------------------------
    for i_node in range(socp.n_shooting):
        constraint_value = (
            socp.nlp[0]
            .g[8]
            .function[i_node](
                duration,
                dt,
                x_multi_thread[:, i_node],
                vertcat(tau_sol[:, i_node], tau_sol[:, i_node + 1]),
                p_sol,
                vertcat(a_sol[:, i_node], a_sol[:, i_node + 1]),
            )
        )
        np.testing.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)

    # Covariance continuity --------------------------------------------------------------------
    constraint_value = (
        socp.nlp[0]
        .g[9]
        .function[0](
            duration,
            dt,
            x_multi_thread,
            vertcat(tau_sol[:, 1:], tau_sol[:, :-1]),
            p_sol,
            vertcat(a_sol[:, 1:], a_sol[:, :-1]),
        )
    )
    np.testing.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)

    # States continuity --------------------------------------------------------------------
    constraint_value = (
        socp.nlp[0]
        .g_internal[0]
        .function[0](
            duration,
            dt,
            x_multi_thread,
            vertcat(tau_sol[:, 1:], tau_sol[:, :-1]),
            p_sol,
            vertcat(a_sol[:, 1:], a_sol[:, :-1]),
        )
    )
    np.testing.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)

    # First collocation state is equal to the states at node
    for i_node in range(socp.n_shooting):
        constraint_value = (
            socp.nlp[0]
            .g_internal[1]
            .function[i_node](
                duration,
                dt,
                x_multi_thread[:, i_node],
                vertcat(tau_sol[:, i_node], tau_sol[:, i_node + 1]),
                p_sol,
                vertcat(a_sol[:, i_node], a_sol[:, i_node + 1]),
            )
        )
        np.testing.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)


@pytest.mark.parametrize("use_sx", [False, True])
def test_obstacle_avoidance_direct_collocation(use_sx: bool):
    from bioptim.examples.stochastic_optimal_control import obstacle_avoidance_direct_collocation as ocp_module

    polynomial_degree = 3
    n_shooting = 10

    q_init = np.zeros((2, (polynomial_degree + 2) * n_shooting + 1))
    zq_init = ocp_module.initialize_circle((polynomial_degree + 1) * n_shooting + 1)
    for i in range(n_shooting + 1):
        j = i * (polynomial_degree + 1)
        k = i * (polynomial_degree + 2)
        q_init[:, k] = zq_init[:, j]
        q_init[:, k + 1 : k + 1 + (polynomial_degree + 1)] = zq_init[:, j : j + (polynomial_degree + 1)]

    ocp = ocp_module.prepare_socp(
        final_time=4,
        n_shooting=n_shooting,
        polynomial_degree=polynomial_degree,
        motor_noise_magnitude=np.array([1, 1]),
        q_init=q_init,
        is_stochastic=True,
        is_robustified=True,
        socp_type=SocpType.COLLOCATION(polynomial_degree=polynomial_degree, method="legendre"),
        use_sx=use_sx,
    )

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(4)
    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 4.6220107868123605)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (1043, 1))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    algebraic_states = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)

    q, qdot = states["q"], states["qdot"]
    u = controls["u"]
    m, cov = algebraic_states["m"], algebraic_states["cov"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([-1.07999204e-27, 2.94926475e00]))
    np.testing.assert_almost_equal(q[:, -1], np.array([-3.76592146e-26, 2.94926475e00]))
    np.testing.assert_almost_equal(qdot[:, 0], np.array([3.59388215, 0.49607651]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([3.59388215, 0.49607651]))

    np.testing.assert_almost_equal(u[:, 0], np.array([2.2568354, 1.69720657]))
    np.testing.assert_almost_equal(u[:, -1], np.array([0.82746288, 2.89042815]))

    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                1.00000000e00,
                -5.05457090e-25,
                -3.45225516e-23,
                4.63567667e-24,
                -2.07762174e-24,
                1.00000000e00,
                5.85505404e-24,
                2.11044910e-24,
                5.35541145e-25,
                -7.33375346e-25,
                1.00000000e00,
                3.31004423e-24,
                6.69132819e-25,
                -1.55199996e-25,
                1.61445742e-24,
                1.00000000e00,
                1.90797247e-01,
                -1.19090552e-02,
                -3.23045118e-01,
                -8.36867760e-02,
                -1.29812817e-02,
                1.69927215e-01,
                -9.02323302e-02,
                -4.15440327e-01,
                2.91358598e-02,
                4.62429927e-03,
                -4.04540496e-02,
                2.59478026e-03,
                5.65168256e-03,
                4.62998816e-02,
                5.73943076e-03,
                -3.07383562e-02,
                3.91343262e-01,
                -6.89506402e-03,
                -4.87839314e-01,
                -8.10220212e-02,
                -7.02994760e-03,
                3.85606978e-01,
                -8.33694095e-02,
                -5.61696657e-01,
                4.84320277e-02,
                7.51042245e-03,
                4.20836460e-02,
                4.20298027e-02,
                7.79698790e-03,
                5.92538743e-02,
                4.52842640e-02,
                9.08680212e-02,
                2.76261710e-01,
                9.59731386e-05,
                -1.11028293e-01,
                -7.03012679e-03,
                6.11634134e-05,
                2.76243341e-01,
                -6.74241321e-03,
                -1.14566661e-01,
                1.09070369e-02,
                7.09878476e-04,
                1.98625775e-01,
                1.83359034e-02,
                7.31642248e-04,
                1.11477554e-02,
                1.81224176e-02,
                2.12172685e-01,
            ]
        ),
        decimal=6,
    )

    np.testing.assert_almost_equal(
        cov[:, -1],
        np.array(
            [
                0.00266764,
                -0.0005587,
                0.00241239,
                -0.00088205,
                -0.0005587,
                0.00134316,
                -0.00048081,
                0.00673894,
                0.00241239,
                -0.00048081,
                -0.00324733,
                -0.00175754,
                -0.00088205,
                0.00673894,
                -0.00175754,
                0.02038775,
            ]
        ),
        decimal=6,
    )
