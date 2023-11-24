import os
from sys import platform

import numpy as np
from casadi import DM, vertcat
from bioptim import Solver, SocpType


def test_arm_reaching_torque_driven_collocations():
    from bioptim.examples.stochastic_optimal_control import arm_reaching_torque_driven_collocations as ocp_module

    if platform != "linux":
        return

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
    states, controls, stochastic_variables = (
        sol.states,
        sol.controls,
        sol.stochastic_variables,
    )
    q, qdot = states["q"], states["qdot"]
    tau = controls["tau"]
    k, ref, m, cov = (
        stochastic_variables["k"],
        stochastic_variables["ref"],
        stochastic_variables["m"],
        stochastic_variables["cov"],
    )

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.9256103, 1.29037205]))
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

    np.testing.assert_almost_equal(tau[:, 0], np.array([1.72235954, -0.90041542]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-1.64870266, 1.08550928]))

    np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412560e-01, 0, 0]))
    # np.testing.assert_almost_equal(
    #     m[:, 0],
    #     np.array(
    #         [
    #             1.00000000e00,
    #             6.56332273e-28,
    #             1.74486171e-28,
    #             2.01948392e-28,
    #             -3.89093870e-28,
    #             1.00000000e00,
    #             1.56788574e-28,
    #             -2.17796617e-28,
    #             5.05560847e-28,
    #             2.85012070e-27,
    #             1.00000000e00,
    #             1.54141267e-28,
    #             5.57682185e-27,
    #             -5.63150297e-27,
    #             -3.45241276e-28,
    #             1.00000000e00,
    #             2.73326455e-01,
    #             8.82635686e-04,
    #             -1.12426658e-01,
    #             -4.65696926e-02,
    #             -7.91605355e-04,
    #             2.93109236e-01,
    #             -7.62158205e-03,
    #             2.52356513e-01,
    #             2.12258920e-02,
    #             -6.36373977e-03,
    #             1.97796394e-01,
    #             -1.13276740e-01,
    #             8.12826441e-04,
    #             1.12560995e-02,
    #             1.27484494e-02,
    #             4.62100410e-02,
    #             4.41345368e-01,
    #             -2.20958405e-03,
    #             -1.28278134e-01,
    #             -1.23557656e-01,
    #             -3.31136999e-04,
    #             4.53208614e-01,
    #             -9.74325601e-03,
    #             2.94725085e-01,
    #             2.03928892e-02,
    #             -4.37542245e-03,
    #             3.69716585e-01,
    #             -1.56027946e-01,
    #             5.37451463e-04,
    #             1.38558352e-02,
    #             1.82648112e-02,
    #             1.54357099e-01,
    #             2.77666284e-01,
    #             -3.63464736e-04,
    #             -2.16330322e-02,
    #             -3.87026374e-02,
    #             -2.84987409e-05,
    #             2.78133442e-01,
    #             -2.59652367e-03,
    #             5.58736715e-02,
    #             3.07535278e-03,
    #             -2.44509904e-04,
    #             2.67220432e-01,
    #             -3.40252069e-02,
    #             2.94621777e-05,
    #             2.74660013e-03,
    #             4.13178488e-03,
    #             2.19586629e-01,
    #         ]
    #     ),
    #     decimal=2,
    # )

    # np.testing.assert_almost_equal(
    #     cov[:, -2],
    #     np.array(
    #         [
    #             -0.56657318,
    #             -0.57490179,
    #             -0.66005047,
    #             -0.22158913,
    #             -0.57490244,
    #             -0.52722059,
    #             -0.43145661,
    #             -0.36735762,
    #             -0.66004847,
    #             -0.43145651,
    #             -0.40759851,
    #             -0.06068207,
    #             -0.2215913,
    #             -0.36735785,
    #             -0.06068166,
    #             -0.3793242,
    #         ]
    #     ),
    #     decimal=1,
    # )
    # TODO: See the file diff to understand why the values are different (otherwise check with Pariterre)


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

    q_sol = sol_socp.states["q"]
    qdot_sol = sol_socp.states["qdot"]
    tau_sol = sol_socp.controls["tau"]
    k_sol = sol_socp.stochastic_variables["k"]
    ref_sol = sol_socp.stochastic_variables["ref"]
    m_sol = sol_socp.stochastic_variables["m"]
    cov_sol = sol_socp.stochastic_variables["cov"]

    polynomial_degree = socp.nlp[0].ode_solver.polynomial_degree

    # Constraint values
    x_opt = vertcat(q_sol, qdot_sol)
    x_sol = np.zeros((x_opt.shape[0], polynomial_degree + 2, socp.n_shooting))
    for i_node in range(socp.n_shooting):
        x_sol[:, :, i_node] = x_opt[:, i_node * (polynomial_degree + 2):(i_node + 1) * (polynomial_degree + 2)]
    s_sol = vertcat(k_sol, ref_sol, m_sol, cov_sol)

    # Initial posture
    shoulder_pos_initial = 0.349065850398866
    elbow_pos_initial = 2.245867726451909
    constraint_value = socp.nlp[0].g[0].function[0](0,
                                                    x_sol[:, :, 0].flatten(order="F"),
                                                    tau_sol[:, 0],
                                                    [],
                                                    s_sol[:, 0],
                                                    )
    np.testing.assert_almost_equal(constraint_value[0], shoulder_pos_initial, decimal=6)
    np.testing.assert_almost_equal(constraint_value[1], elbow_pos_initial, decimal=6)

    # Initial and final velocities
    constraint_value = socp.nlp[0].g[1].function[0](0,
                                                    x_sol[:, :, 0].flatten(order="F"),
                                                    tau_sol[:, 0],
                                                    [],
                                                    s_sol[:, 0],
                                                    )
    np.testing.assert_almost_equal(constraint_value[0], 0, decimal=6)
    np.testing.assert_almost_equal(constraint_value[1], 0, decimal=6)
    constraint_value = socp.nlp[0].g[2].function[-1](0,
                                                    x_opt[:, -1],
                                                    tau_sol[:, -1],
                                                    [],
                                                    s_sol[:, -1],
                                                    )
    np.testing.assert_almost_equal(constraint_value[0], 0, decimal=6)
    np.testing.assert_almost_equal(constraint_value[1], 0, decimal=6)

    # Hand final marker position
    constraint_value = socp.nlp[0].g[4].function[-1](0,
                                                    x_opt[:, -1],
                                                    tau_sol[:, -1],
                                                    [],
                                                    s_sol[:, -1],
                                                    )
    np.testing.assert_almost_equal(constraint_value[0], hand_final_position[0], decimal=6)
    np.testing.assert_almost_equal(constraint_value[1], hand_final_position[1], decimal=6)

    # Reference equals mean sensory input
    for i_node in range(socp.n_shooting):
        constraint_value = socp.nlp[0].g[7].function[i_node](0,
                                                             x_sol[:, :, i_node].flatten(order="F"),
                                                             tau_sol[:, i_node],
                                                             [],
                                                             s_sol[:, i_node],
                                                             )
        np.testing.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)

    # Constraint on M
    for i_node in range(socp.n_shooting):
        constraint_value = socp.nlp[0].g[8].function[i_node](0,
                                                             x_sol[:, :, i_node].flatten(order="F"),
                                                             tau_sol[:, i_node],
                                                             [],
                                                             s_sol[:, i_node],
                                                             )
        np.testing.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)

    # # Covariance continuity
    # x_multi_thread = np.zeros((socp.nlp[0].states.shape * (polynomial_degree + 3), socp.nlp[0].ns))
    # for i_node in range(socp.nlp[0].ns):
    #     for i_state in range(socp.nlp[0].states.shape):
    #         x_multi_thread[i_state, i_node] = x_sol[i_state, 0, i_node]
    #         for i_coll in range(1, polynomial_degree + 3):
    #             x_multi_thread[i_coll*socp.nlp[0].states.shape + i_state, i_node] = x_sol[i_state, i_coll-1, i_node]
    # constraint_value = socp.nlp[0].g[9].function[0](0,
    #                           x_multi_thread,
    #                           vertcat(tau_sol[:, :-1], tau_sol[:, 1:]),
    #                           [],
    #                           vertcat(s_sol[:, :-1], s_sol[:, 1:]),
    #                           )
    # np.testing.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)
    #
    # # States continuity
    # constraint_value = socp.nlp[0].g_internal[0].function[0](0,
    #                                                          x_multi_thread,
    #                                                          vertcat(tau_sol[:, :-1],
    #                                                                      tau_sol[:, 1:]),
    #                                                          [],
    #                                                          vertcat(s_sol[:, :-1], s_sol[:, 1:]),
    #                                                          )
    # np.testing.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)

    # First collocation state is equal to the states at node
    for i_node in range(socp.n_shooting):
        constraint_value = socp.nlp[0].g_internal[1].function[i_node](0,
                                  x_sol[:, :, i_node].flatten(order="F"),
                                  tau_sol[:, i_node],
                                  [],
                                  s_sol[:, i_node],
                                  )
        np.testing.assert_almost_equal(constraint_value, np.zeros(constraint_value.shape), decimal=6)


def test_obstacle_avoidance_direct_collocation():
    from bioptim.examples.stochastic_optimal_control import obstacle_avoidance_direct_collocation as ocp_module

    # if platform != "linux":
    #     return

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
        is_sotchastic=True,
        is_robustified=True,
        socp_type=SocpType.COLLOCATION(polynomial_degree=polynomial_degree, method="legendre"),
    )

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(4)
    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 4.587065067031554)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (1043, 1))

    # Check some of the results
    states, controls, stochastic_variables = (
        sol.states,
        sol.controls,
        sol.stochastic_variables,
    )
    q, qdot = states["q"], states["qdot"]
    u = controls["u"]
    m, cov = (
        stochastic_variables["m"],
        stochastic_variables["cov"],
    )

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.0, 2.91660270e00]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.0, 2.91660270e00]))
    np.testing.assert_almost_equal(qdot[:, 0], np.array([4.59876163, 0.33406115]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([4.59876163, 0.33406115]))

    np.testing.assert_almost_equal(u[:, 0], np.array([3.94130314, 0.50752995]))
    np.testing.assert_almost_equal(u[:, -2], np.array([1.37640701, 2.78054156]))

    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                1.00000000e00,
                -1.05389293e-23,
                -9.29903240e-24,
                1.00382361e-23,
                -1.64466833e-23,
                1.00000000e00,
                1.21492152e-24,
                -3.15104115e-23,
                -6.68416587e-25,
                -6.00029062e-24,
                1.00000000e00,
                1.99489733e-23,
                -1.16322274e-24,
                -2.03253417e-24,
                -3.00499207e-24,
                1.00000000e00,
                2.19527862e-01,
                -1.88588087e-02,
                -2.00283989e-01,
                -8.03404360e-02,
                -1.99327784e-02,
                2.02962627e-01,
                -8.39758964e-02,
                -2.49822789e-01,
                1.76793622e-02,
                5.30096916e-03,
                -6.35628572e-03,
                -1.01527618e-02,
                6.21147642e-03,
                2.87692596e-02,
                -1.06499714e-02,
                -1.48244735e-02,
                4.01184050e-01,
                -1.20760665e-02,
                -3.47575458e-01,
                -1.01031369e-01,
                -1.22801502e-02,
                3.94781689e-01,
                -1.03912381e-01,
                -4.08950331e-01,
                3.31437788e-02,
                9.65931210e-03,
                1.64098610e-03,
                3.61379227e-02,
                9.94099379e-03,
                4.10555191e-02,
                3.89631730e-02,
                2.71848362e-02,
                2.74709609e-01,
                -6.03467730e-05,
                -1.00613832e-01,
                -1.27941917e-02,
                -9.52485792e-05,
                2.74478998e-01,
                -1.23522568e-02,
                -1.07746467e-01,
                1.00776666e-02,
                1.25778066e-03,
                1.65876475e-01,
                2.50629520e-02,
                1.28718848e-03,
                1.07109173e-02,
                2.48728130e-02,
                1.81242999e-01,
            ]
        ),
        decimal=6,
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                0.00440214,
                -0.00021687,
                0.00470812,
                -0.00133034,
                -0.00021687,
                0.00214526,
                -0.00098746,
                0.00142654,
                0.00470812,
                -0.00098746,
                0.02155766,
                -0.00941652,
                -0.00133034,
                0.00142654,
                -0.00941652,
                0.00335482,
            ]
        ),
        decimal=6,
    )
