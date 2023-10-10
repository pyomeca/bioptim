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
    np.testing.assert_almost_equal(f[0, 0], 426.84572115797613)

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
        np.array([0.00244702, -0.00785471,  0.00078658, -0.0024329 , -0.00430568,
        0.01025431, -0.00152375,  0.00513085]),
    )
    np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412560e-01, 0, 0]))
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                1.00000000e+00, -3.77895162e-26, 3.59168370e-27, -2.25026518e-26,
                -5.83841324e-27, 1.00000000e+00, -2.50561120e-27, -8.05614584e-28,
                1.07013354e-26, -6.02530693e-26, 1.00000000e+00, -3.62999593e-27,
                1.60986410e-26, -7.26013836e-27, -2.38831726e-27, 1.00000000e+00,
                2.77785436e-01, -1.61590034e-05, 3.62208182e-04, -8.63370905e-04,
                7.15686037e-04, 2.88717196e-01, 9.38250307e-03, 2.62016683e-01,
                2.46063725e-02, -5.17046167e-04, 2.73872502e-01, -1.77384799e-02,
                -1.39218226e-04, 2.58152411e-02, -4.97906442e-03, 3.09997389e-01,
                4.44455666e-01, -2.73664838e-05, 5.50037344e-04, -1.36292920e-03,
                2.11475780e-04, 4.50265274e-01, 3.26727462e-03, 2.43822253e-01,
                2.20829819e-02, -4.87368979e-04, 4.37275273e-01, -2.27485090e-02,
                -1.34603763e-04, 2.27760235e-02, -6.51257167e-03, 4.69709775e-01,
                2.77778393e-01, -1.46916530e-06, 1.09375768e-04, -2.70321697e-04,
                -1.09393291e-05, 2.77997454e-01, -6.12964959e-04, 3.62631905e-02,
                3.12128033e-03, -2.34425652e-05, 2.76208157e-01, -4.16028732e-03,
                -7.75901119e-06, 3.14781501e-03, -1.23382279e-03, 2.80812944e-01,
            ]
        ),
        decimal=5,
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                -0.29205426, 0.61848027, -1.90174829, -1.85503884, 0.61848027,
                0.08515987, -0.52547684, -0.59414631, -1.90174828, -0.52547684,
                -1.11687332, -1.26364766, -1.85503884, -0.59414631, -1.26364766,
                -2.97359533,
            ]
        ),
        decimal=4,
    )
