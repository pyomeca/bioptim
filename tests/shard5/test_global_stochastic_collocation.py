import os
from sys import platform

import numpy as np
from casadi import DM, vertcat
from bioptim import Solver


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
    np.testing.assert_almost_equal(tau[:, -2], np.array([-1.64870266, 1.08550928]))

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array([-0.4667972 , -3.26899096, -0.1631994 , -1.9435918 ,  3.00992629,
        3.20946796,  0.28313214,  3.40857073]),
        decimal=5,
    )
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
        decimal=5,
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
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
        decimal=4,
    )
