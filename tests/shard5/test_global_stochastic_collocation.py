import os
from sys import platform

import numpy as np
from casadi import DM, vertcat
from bioptim import Solver, SocpType


def test_arm_reaching_torque_driven_collocations():
    from bioptim.examples.stochastic_optimal_control import arm_reaching_torque_driven_collocations as ocp_module

    # if platform != "linux":
    #     return

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
    np.testing.assert_almost_equal(f[0, 0], 426.8457209111154)

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
    np.testing.assert_almost_equal(tau[:, -1], np.array([-1.64870266, 1.08550928]))

    np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412560e-01, 0, 0]))
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                1.00000000e00,
                1.10362271e-24,
                2.22954931e-28,
                -1.20515105e-27,
                -4.82198156e-25,
                1.00000000e00,
                -8.11748571e-29,
                -1.83751963e-28,
                4.94557578e-25,
                -1.78093812e-25,
                1.00000000e00,
                -2.66499977e-27,
                -3.86066682e-25,
                6.67095096e-26,
                -1.34508637e-27,
                1.00000000e00,
                2.73268581e-01,
                8.12760759e-04,
                -1.13937433e-01,
                -4.80582813e-02,
                -6.82535380e-04,
                2.92927116e-01,
                -4.48357025e-03,
                2.46779124e-01,
                2.11544998e-02,
                -6.30362089e-03,
                1.96229280e-01,
                -1.11381918e-01,
                8.36502545e-04,
                1.10428331e-02,
                1.30538280e-02,
                4.38185419e-02,
                4.41297671e-01,
                -2.25568192e-03,
                -1.30256542e-01,
                -1.25440894e-01,
                -2.53765683e-04,
                4.53128027e-01,
                -6.27183235e-03,
                2.90336583e-01,
                2.03549193e-02,
                -4.35582809e-03,
                3.68201657e-01,
                -1.54792782e-01,
                5.56316902e-04,
                1.36916324e-02,
                1.88617201e-02,
                1.49541324e-01,
                2.77664470e-01,
                -3.70932151e-04,
                -2.19982988e-02,
                -3.92648307e-02,
                -2.51133315e-05,
                2.78132403e-01,
                -2.01389570e-03,
                5.55215545e-02,
                3.07424160e-03,
                -2.45969266e-04,
                2.67004519e-01,
                -3.40457592e-02,
                3.07722398e-05,
                2.73583597e-03,
                4.29990529e-03,
                2.18202901e-01,
            ]
        ),
        decimal=2,
    )

    np.testing.assert_almost_equal(
        cov[:, -1],
        np.array(
            [
                -0.57472415,
                -0.58086399,
                -0.65152237,
                -0.20929523,
                -0.5808627,
                -0.54536228,
                -0.42680683,
                -0.36921097,
                -0.65154088,
                -0.42679628,
                -0.41733866,
                -0.04060819,
                -0.20928018,
                -0.36921833,
                -0.04062543,
                -0.37958588,
            ]
        ),
        decimal=2,
    )


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
    np.testing.assert_almost_equal(u[:, -1], np.array([1.37640701, 2.78054156]))

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
