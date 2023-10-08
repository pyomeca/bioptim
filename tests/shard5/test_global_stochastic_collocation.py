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
    np.testing.assert_almost_equal(f[0, 0], 423.5304701565359)

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

    np.testing.assert_almost_equal(tau[:, 0], np.array([1.73581143, -0.9298179]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-1.64142393, 1.05989069]))

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array([0.00292193, 0.10196103, -0.01810361, 0.13623824, 0.01848452, 0.02874408, 0.0149202, 0.0167177]),
    )
    np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412560e-01, 0, 0]))
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                1.00000000e00,
                -3.15544362e-30,
                7.87897940e-31,
                -1.55884770e-30,
                -7.91172021e-31,
                1.00000000e00,
                7.82665204e-31,
                1.19022471e-31,
                -1.18196728e-30,
                0.00000000e00,
                1.00000000e00,
                -4.73316543e-30,
                1.57695144e-30,
                1.26217745e-29,
                1.23259516e-32,
                1.00000000e00,
                2.77877331e-01,
                -6.06720358e-04,
                2.39734559e-03,
                -1.54194480e-02,
                9.33581421e-04,
                2.87786926e-01,
                1.42239691e-02,
                2.40267334e-01,
                2.45708119e-02,
                -6.05041259e-04,
                2.73030823e-01,
                -2.00511500e-02,
                -1.82493750e-04,
                2.57128572e-02,
                -5.94326260e-03,
                3.07397177e-01,
                4.44496262e-01,
                -3.57336301e-04,
                2.21384942e-03,
                -1.53138266e-02,
                3.25584752e-04,
                4.49809252e-01,
                7.75405753e-03,
                2.25294053e-01,
                2.20605894e-02,
                -5.30502740e-04,
                4.36357737e-01,
                -2.46479500e-02,
                -1.60589868e-04,
                2.27331013e-02,
                -7.55050119e-03,
                4.67862790e-01,
                2.77779858e-01,
                -1.38811733e-05,
                3.27865873e-04,
                -2.38202953e-03,
                -7.41621448e-06,
                2.77983763e-01,
                3.25142088e-05,
                3.37975188e-02,
                3.12034906e-03,
                -2.49462228e-05,
                2.76052053e-01,
                -4.41070308e-03,
                -8.76593285e-06,
                3.14670265e-03,
                -1.40592443e-03,
                2.80616267e-01,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                -0.20373395,
                -0.10151324,
                -0.02788844,
                0.02434424,
                -0.10151324,
                -0.15651666,
                0.02849463,
                0.01067661,
                -0.02788844,
                0.02849463,
                -0.01658184,
                -0.02042144,
                0.02434424,
                0.01067661,
                -0.02042144,
                0.01397916,
            ]
        ),
    )
