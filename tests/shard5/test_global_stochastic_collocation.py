import os
import pytest
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
    np.testing.assert_almost_equal(f[0, 0], 426.82741618989667)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (426, 1))

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

    np.testing.assert_almost_equal(tau[:, 0], np.array([1.73031277, -0.91599583]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-1.64310999, 1.06734485]))

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array([0.0175521, 0.0175521, 0.0245837, 0.0245837, 0.01278656, 0.01278656, 0.01431021, 0.01431021]),
    )
    np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412560e-01, 0, 0]))
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                1.00000000e00,
                -4.68755941e-28,
                3.11410731e-26,
                3.14622973e-25,
                2.06287850e-03,
                1.05023845e00,
                1.07351243e-02,
                1.07761025e00,
                9.99905817e-02,
                -1.52335846e-03,
                9.85797229e-01,
                -5.77211779e-02,
                -6.60261979e-04,
                1.05793581e-01,
                -2.22658019e-02,
                1.14354394e00,
                -2.77777778e-01,
                3.23652527e-29,
                -1.12958719e-27,
                -6.70184671e-26,
                -3.79685002e-04,
                -2.88884694e-01,
                -1.48813445e-03,
                -2.66793612e-01,
                -2.45775389e-02,
                4.61321297e-04,
                -2.73151208e-01,
                1.64247278e-02,
                1.78686644e-04,
                -2.58900136e-02,
                6.06169081e-03,
                -3.11888534e-01,
                -4.44444444e-01,
                1.81658527e-29,
                -7.45873224e-27,
                -3.36526337e-26,
                -3.78694281e-05,
                -4.50359586e-01,
                3.87711510e-03,
                -2.48211335e-01,
                -2.20660682e-02,
                4.59860530e-04,
                -4.36558093e-01,
                2.16153102e-02,
                1.54111460e-04,
                -2.28135913e-02,
                7.39476052e-03,
                -4.71318552e-01,
                -2.77777778e-01,
                -6.57907299e-30,
                2.04745439e-27,
                -5.88445631e-27,
                1.69372039e-05,
                -2.78001507e-01,
                1.63750462e-03,
                -3.69137642e-02,
                -3.12059215e-03,
                2.25020445e-05,
                -2.76094043e-01,
                4.00312361e-03,
                8.41274934e-06,
                -3.14904548e-03,
                1.34665643e-03,
                -2.81028066e-01,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                -0.00122028,
                -0.00104199,
                -0.00664454,
                -0.00431037,
                -0.00104199,
                -0.23710625,
                -0.00374855,
                -0.0026406,
                -0.00664454,
                -0.00374855,
                -0.02103248,
                -0.01061616,
                -0.00431037,
                -0.0026406,
                -0.01061616,
                -0.08671307,
            ]
        ),
    )
