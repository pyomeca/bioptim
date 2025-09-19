import platform

import numpy as np
import numpy.testing as npt
import pytest
from casadi import DM, vertcat, horzcat

from bioptim import Solver, SocpType, SolutionMerge, PenaltyHelpers, SolutionIntegrator, StochasticBioModel
from ..utils import TestUtils


@pytest.mark.parametrize("use_sx", [False, True])
def test_arm_reaching_torque_driven_collocations(use_sx: bool):
    from bioptim.examples.toy_examples.stochastic_optimal_control import (
        arm_reaching_torque_driven_collocations as ocp_module,
    )

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

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_socp(
        biorbd_model_path=bioptim_folder + "/examples/models/LeuvenArmModel.bioMod",
        final_time=final_time,
        n_shooting=n_shooting,
        polynomial_degree=3,
        hand_final_position=hand_final_position,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        use_sx=use_sx,
    )

    # Solver parameters
    solver = Solver.IPOPT()
    solver.set_nlp_scaling_method("none")

    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 433.119929307444)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (442, 1))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    algebraic_states = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)

    q, qdot = states["q"], states["qdot"]
    tau = controls["tau"]
    k, ref, m, cov = controls["k"], controls["ref"], algebraic_states["m"], controls["cov"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
    npt.assert_almost_equal(q[:, -1], np.array([0.9256103, 1.29037205]))
    npt.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
    npt.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

    npt.assert_almost_equal(tau[:, 0], np.array([1.73918356, -1.0035866]))
    npt.assert_almost_equal(tau[:, -2], np.array([-1.672167, 0.91772376]))

    npt.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412560e-01, 0, 0]))

    npt.assert_almost_equal(
        m[:10, 1],
        np.array(
            [
                1.00000000e00,
                0.0,
                0.0,
                0.0,
                0.0,
                1.00000000e00,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )

    # TODO: cov is still too sensitive to be properly tested, we need to test it otherwise

    # Test the automatic initialization of the stochastic variables
    socp = ocp_module.prepare_socp(
        biorbd_model_path=bioptim_folder + "/examples/models/LeuvenArmModel.bioMod",
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
    solver = Solver.IPOPT()
    solver.set_nlp_scaling_method("none")
    solver.set_maximum_iterations(0)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)

    sol_socp = socp.solve(solver)

    # Check some of the results
    states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
    q_sol, qdot_sol = states["q"], states["qdot"]
    states_sol = np.zeros((4, 5, 5))
    for i in range(4):
        states_sol[:, :, i] = sol_socp.decision_states(to_merge=SolutionMerge.KEYS)[i]
    states_sol[:, 0, 4] = np.reshape(sol_socp.decision_states(to_merge=SolutionMerge.KEYS)[-1], (-1,))

    controls_sol = sol_socp.decision_controls(to_merge=SolutionMerge.ALL)

    algebraic_sol = np.zeros((16, 5, 5))
    for i in range(4):
        algebraic_sol[:, :, i] = sol_socp.decision_algebraic_states(to_merge=SolutionMerge.KEYS)[i]
    algebraic_sol[:, 0, 4] = np.reshape(sol_socp.decision_algebraic_states(to_merge=SolutionMerge.KEYS)[-1], (-1,))

    duration = sol_socp.decision_time()[-1]
    dt = duration / n_shooting
    p_sol = vertcat(ocp.nlp[0].model.motor_noise_magnitude, ocp.nlp[0].model.sensory_noise_magnitude)
    polynomial_degree = socp.nlp[0].dynamics_type.ode_solver.polynomial_degree

    # Constraint values
    x_opt = vertcat(q_sol, qdot_sol)
    x_sol = np.zeros((x_opt.shape[0], polynomial_degree + 2, socp.n_shooting))
    for i_node in range(socp.n_shooting):
        x_sol[:, :, i_node] = x_opt[:, i_node * (polynomial_degree + 2) : (i_node + 1) * (polynomial_degree + 2)]

    x_multi_thread = np.zeros((socp.nlp[0].states.shape * (polynomial_degree + 3), socp.nlp[0].ns))
    for i_node in range(socp.nlp[0].ns):
        for i_state in range(socp.nlp[0].states.shape):
            x_multi_thread[i_state, i_node] = x_sol[i_state, 0, i_node]
            for i_coll in range(1, polynomial_degree + 3):
                x_multi_thread[i_coll * socp.nlp[0].states.shape + i_state, i_node] = x_sol[i_state, i_coll - 1, i_node]

    # Initial posture
    penalty = socp.nlp[0].g[0]
    x = PenaltyHelpers.states(
        penalty,
        0,
        lambda p_idx, n_idx, sn_idx: states_sol[:, sn_idx.index(), n_idx],
    )
    u = PenaltyHelpers.controls(
        penalty,
        0,
        lambda p_idx, n_idx, sn_idx: controls_sol[:, n_idx],
    )
    a = PenaltyHelpers.states(
        penalty,
        0,
        lambda p_idx, n_idx, sn_idx: algebraic_sol[:, sn_idx.index(), n_idx],
    )
    shoulder_pos_initial = 0.349065850398866
    elbow_pos_initial = 2.245867726451909

    constraint_value = penalty.function[0](
        duration,
        dt,
        x,
        u,
        p_sol,
        a,
        [],
    )
    npt.assert_almost_equal(constraint_value[0], shoulder_pos_initial, decimal=6)
    npt.assert_almost_equal(constraint_value[1], elbow_pos_initial, decimal=6)

    # Initial and final velocities
    penalty = socp.nlp[0].g[1]
    x = PenaltyHelpers.states(
        penalty,
        0,
        lambda p_idx, n_idx, sn_idx: states_sol[:, sn_idx.index(), n_idx],
    )
    u = PenaltyHelpers.controls(
        penalty,
        0,
        lambda p_idx, n_idx, sn_idx: controls_sol[:, n_idx],
    )
    a = PenaltyHelpers.states(
        penalty,
        0,
        lambda p_idx, n_idx, sn_idx: algebraic_sol[:, sn_idx.index(), n_idx],
    )
    constraint_value = penalty.function[0](
        duration,
        dt,
        x,
        u,
        p_sol,
        a,
        [],
    )
    npt.assert_almost_equal(constraint_value[0], 0, decimal=6)
    npt.assert_almost_equal(constraint_value[1], 0, decimal=6)

    penalty = socp.nlp[0].g[2]
    x = states_sol[:, 0, -1]
    u = controls_sol[:, -1]
    a = algebraic_sol[:, 0, -1]
    constraint_value = penalty.function[-1](
        duration,
        dt,
        x,
        u,
        p_sol,
        a,
        [],
    )
    npt.assert_almost_equal(constraint_value[0], 0, decimal=6)
    npt.assert_almost_equal(constraint_value[1], 0, decimal=6)

    # Hand final marker position
    penalty = socp.nlp[0].g[4]
    x = states_sol[:, 0, -1]
    u = controls_sol[:, -1]
    a = algebraic_sol[:, 0, -1]
    constraint_value = penalty.function[-1](
        duration,
        dt,
        x,
        u,
        p_sol,
        a,
        [],
    )
    npt.assert_almost_equal(constraint_value[0], hand_final_position[0], decimal=6)
    npt.assert_almost_equal(constraint_value[1], hand_final_position[1], decimal=6)

    # Reference equals mean sensory input
    penalty = socp.nlp[0].g[7]
    for i_node in range(socp.n_shooting):
        x = PenaltyHelpers.states(
            penalty,
            0,
            lambda p_idx, n_idx, sn_idx: states_sol[:, sn_idx.index(), n_idx],
        )
        u = PenaltyHelpers.controls(
            penalty,
            0,
            lambda p_idx, n_idx, sn_idx: controls_sol[:, n_idx],
        )
        a = PenaltyHelpers.states(
            penalty,
            0,
            lambda p_idx, n_idx, sn_idx: algebraic_sol[:, sn_idx.index(), n_idx],
        )
        constraint_value = penalty.function[i_node](
            duration,
            dt,
            x,
            u,
            p_sol,
            a,
            [],
        )
        npt.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)

    # Constraint on M --------------------------------------------------------------------
    penalty = socp.nlp[0].g[8]
    for i_node in range(socp.n_shooting):
        x = PenaltyHelpers.states(
            penalty,
            i_node,
            lambda p_idx, n_idx, sn_idx: states_sol[:, sn_idx.index(), n_idx],
        )
        u = PenaltyHelpers.controls(
            penalty,
            i_node,
            lambda p_idx, n_idx, sn_idx: controls_sol[:, n_idx],
        )
        a = PenaltyHelpers.states(
            penalty,
            i_node,
            lambda p_idx, n_idx, sn_idx: algebraic_sol[:, sn_idx.index(), n_idx],
        )
        constraint_value = penalty.function[i_node](
            duration,
            dt,
            x,
            u,
            p_sol,
            a,
            [],
        )
        npt.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)

    # Covariance continuity --------------------------------------------------------------------
    penalty = socp.nlp[0].g[9]
    for i_node in range(socp.n_shooting):
        x = PenaltyHelpers.states(
            penalty,
            i_node,
            lambda p_idx, n_idx, sn_idx: states_sol[:, sn_idx.index(), n_idx],
        )
        u = PenaltyHelpers.controls(
            penalty,
            i_node,
            lambda p_idx, n_idx, sn_idx: controls_sol[:, n_idx],
        )
        a = PenaltyHelpers.states(
            penalty,
            i_node,
            lambda p_idx, n_idx, sn_idx: algebraic_sol[:, sn_idx.index(), n_idx],
        )

        constraint_value = penalty.function[0](
            duration,
            dt,
            x,
            u,
            p_sol,
            a,
            [],
        )
        npt.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)

    # States continuity --------------------------------------------------------------------
    penalty = socp.nlp[0].g_internal[0]
    for i_node in range(socp.n_shooting):
        x = PenaltyHelpers.states(
            penalty,
            i_node,
            lambda p_idx, n_idx, sn_idx: states_sol[:, sn_idx.index(), n_idx],
        )
        u = PenaltyHelpers.controls(
            penalty,
            i_node,
            lambda p_idx, n_idx, sn_idx: controls_sol[:, n_idx],
        )
        a = PenaltyHelpers.states(
            penalty,
            i_node,
            lambda p_idx, n_idx, sn_idx: algebraic_sol[:, sn_idx.index(), n_idx],
        )
        constraint_value = penalty.function[0](
            duration,
            dt,
            x,
            u,
            p_sol,
            a,
            [],
        )
        npt.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)

    # First collocation state is equal to the states at node
    penalty = socp.nlp[0].g_internal[1]
    for i_node in range(socp.n_shooting):
        x = PenaltyHelpers.states(
            penalty,
            i_node,
            lambda p_idx, n_idx, sn_idx: states_sol[:, sn_idx.index(), n_idx],
        )
        u = PenaltyHelpers.controls(
            penalty,
            i_node,
            lambda p_idx, n_idx, sn_idx: controls_sol[:, n_idx],
        )
        a = PenaltyHelpers.states(
            penalty,
            i_node,
            lambda p_idx, n_idx, sn_idx: algebraic_sol[:, sn_idx.index(), n_idx],
        )
        constraint_value = penalty.function[i_node](
            duration,
            dt,
            x,
            u,
            p_sol,
            a,
            [],
        )
        npt.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)


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
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(4)

    # Check the values which will be sent to the solver
    v_size = 1221
    np.random.seed(42)
    TestUtils.compare_ocp_to_solve(
        ocp,
        v=np.ones([v_size, 1]),  # Random values here returns nan for g
        expected_v_f_g=[v_size, 10.01, -170696.19805582374],
        decimal=6,
    )
    if platform.system() == "Windows":
        return

    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 5.831644440290965)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (1043, 1))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    algebraic_states = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)

    q, qdot = states["q"], states["qdot"]
    u = controls["u"]
    m, cov = algebraic_states["m"], controls["cov"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array([9.33415918e-22, 2.98741686e00]))
    npt.assert_almost_equal(q[:, -1], np.array([9.34900539e-22, 2.98741686e00]))
    npt.assert_almost_equal(qdot[:, 0], np.array([2.1919426, 0.48013017]))
    npt.assert_almost_equal(qdot[:, -1], np.array([2.1919426, 0.48013017]))

    npt.assert_almost_equal(u[:, 0], np.array([1.78167973, 1.93951131]))
    npt.assert_almost_equal(u[:, -2], np.array([-0.04051134, 2.84841392]))
    npt.assert_almost_equal(u[:, -1], np.array([0.0, 0.0]))

    m_vector = StochasticBioModel.reshape_to_vector(m[:, [1, 2, 3, 4]])
    npt.assert_almost_equal(
        m_vector,
        np.array(
            [
                1.00000000e00,
                -4.49572966e-22,
                -2.29111163e-18,
                -5.69115647e-19,
                -1.37463144e-18,
                1.00000000e00,
                2.71740231e-20,
                1.88094678e-18,
                2.15236673e-17,
                9.12657816e-22,
                1.00000000e00,
                1.28596672e-18,
                2.59650412e-18,
                -1.98702849e-22,
                1.16174719e-20,
                1.00000000e00,
                8.97683613e-02,
                -9.16539400e-03,
                -1.93148619e-01,
                -4.27328970e-02,
                -9.47788674e-03,
                7.40548692e-02,
                -4.32328089e-02,
                -2.40016559e-01,
                2.23761124e-02,
                2.18125527e-03,
                -1.21607303e-02,
                -2.13744866e-03,
                2.48984982e-03,
                2.80212107e-02,
                -1.78112311e-03,
                -1.51957449e-02,
                2.01865677e-01,
                -6.20640335e-03,
                -2.89187247e-01,
                -5.29601676e-02,
                -6.06943355e-03,
                1.94137289e-01,
                -5.19802435e-02,
                -3.46986013e-01,
                3.21298014e-02,
                3.85369918e-03,
                4.01442349e-02,
                1.71540495e-02,
                3.88746216e-03,
                3.69482073e-02,
                1.76978741e-02,
                5.61188146e-02,
                1.49788124e-01,
                -3.05585700e-05,
                -6.99765523e-02,
                -6.16591796e-03,
                -1.50835457e-05,
                1.49690852e-01,
                -5.61298940e-03,
                -7.60514239e-02,
                7.56174614e-03,
                4.32208391e-04,
                1.11030048e-01,
                9.28078562e-03,
                4.01508193e-04,
                7.98866164e-03,
                8.79013240e-03,
                1.19253753e-01,
            ]
        ),
        decimal=6,
    )

    npt.assert_almost_equal(
        cov[:, -1],
        np.array(
            [
                0.00147778,
                -0.00102866,
                0.00716552,
                -0.00024668,
                -0.00102866,
                0.00210025,
                -0.00015164,
                0.00911479,
                0.00716552,
                -0.00015164,
                -0.0070975,
                -0.01337138,
                -0.00024668,
                0.00911479,
                -0.01337138,
                0.01419299,
            ]
        ),
        decimal=6,
    )

    np.random.seed(42)
    integrated_states = sol.noisy_integrate(integrator=SolutionIntegrator.SCIPY_RK45, to_merge=SolutionMerge.NODES)
    integrated_stated_covariance = np.cov(integrated_states["q"][:, -1, :])
    npt.assert_almost_equal(
        integrated_stated_covariance, np.array([[0.00521863, -0.00169537], [-0.00169537, 0.00616143]]), decimal=6
    )
    npt.assert_almost_equal(
        cov[:, -1].reshape(4, 4)[:2, :2], np.array([[0.00147778, -0.00102866], [-0.00102866, 0.00210025]]), decimal=6
    )
