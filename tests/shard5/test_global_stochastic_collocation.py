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
    solver.set_maximum_iterations(4)
    solver.set_nlp_scaling_method("none")

    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 677.2072748927096)

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
    np.testing.assert_almost_equal(q[:, -1], np.array([0.9256096 , 1.29037459]))
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

    np.testing.assert_almost_equal(tau[:, 0], np.array([1.56117295, -1.14958482]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-1.76391736,  0.9030622]))

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array([-0.08819619, -2.39729859,  0.090945  , -1.47223773,  2.01663046,
        2.51787116, -0.52427039,  0.98901109]),
    )
    np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02,  2.84412560e-01, 0, 0]))
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                1.00000000e+00, -3.09161258e-24, -1.32497598e-26, -1.21456208e-26,
                1.10380743e-24, 1.00000000e+00, -3.13320680e-26, 2.93909144e-27,
                7.06654304e-26, 1.16534127e-24, 1.00000000e+00, -1.76162181e-28,
                3.50620697e-26, -7.43095210e-25, 6.63528939e-26, 1.00000000e+00,
                2.73894230e-01, 3.29519696e-04, -9.43225393e-02, -5.66939602e-02,
                -4.14663798e-03, 2.95213269e-01, -9.86554507e-02, 3.40040244e-01,
                2.24292197e-02, -8.02278460e-03, 2.25322692e-01, -1.65708738e-01,
                7.12933730e-04, 1.61342255e-02, 1.27664506e-02, 1.11368768e-01,
                4.41872874e-01, -2.68688901e-03, -1.01999816e-01, -1.36872896e-01,
                -2.68172526e-03, 4.53441308e-01, -1.10974761e-01, 3.34684631e-01,
                2.10087001e-02, -4.88641177e-03, 3.96094811e-01, -1.86623311e-01,
                3.67075648e-04, 1.73733998e-02, 1.42259590e-02, 2.65841611e-01,
                2.77687308e-01, -2.79066181e-04, -1.62961996e-02, -3.67604531e-02,
                -1.25843386e-04, 2.78077122e-01, -1.95492237e-02, 5.15887622e-02,
                3.09051240e-03, -2.16361407e-04, 2.70987712e-01, -3.32709347e-02,
                1.11782116e-05, 2.94433210e-03, 2.52099767e-03, 2.47605218e-01,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                -0.52268889, -0.56605654, -0.42896801, -0.11572439, -0.56605654,
                -0.49290525, -0.37496778, -0.10287267, -0.42896801, -0.37496778,
                -0.21055025, 0.00145738, -0.11572439, -0.10287267, 0.00145738,
                0.13091722,
            ]
        ),
    )
