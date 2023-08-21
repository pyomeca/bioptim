import os
import pytest
from sys import platform

import numpy as np
from casadi import DM, vertcat
from bioptim import Solver

from bioptim.examples.stochastic_optimal_control.arm_reaching_torque_driven_implicit import ExampleType


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
    np.testing.assert_almost_equal(f[0, 0], 397.99658325589513)

    # detailed cost values
    np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 426.97650484099404)
    np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], -28.979921585098953)
    np.testing.assert_almost_equal(
        f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
    )

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

    np.testing.assert_almost_equal(tau[:, 0], np.array([1.73828351, -0.9215131]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-1.63240552, 1.07053435]))

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array([-0.00449169, 0.2098898, -0.0592484, 0.45527603, 0.00100994, 0.07472448, -0.16921208, 0.23843525]),
    )
    np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412560e-01, 0, 0]))
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                1.00155075e00,
                -6.01473795e-03,
                3.85540646e-02,
                -1.46244160e-01,
                8.32225102e-03,
                1.03271573e00,
                1.49007415e-01,
                6.78896493e-01,
                1.00131640e-01,
                -2.85760196e-03,
                9.89295831e-01,
                -8.78806542e-02,
                2.83216128e-03,
                9.75635186e-02,
                4.99717824e-02,
                9.66887121e-01,
                -2.78155647e-01,
                1.43161737e-03,
                -1.01933525e-02,
                3.80424489e-02,
                -1.74229355e-03,
                -2.85038617e-01,
                -3.51104174e-02,
                -1.69543104e-01,
                -2.46071058e-02,
                7.48964588e-04,
                -2.73946619e-01,
                2.36124721e-02,
                -5.81820507e-04,
                -2.41046121e-02,
                -1.16006769e-02,
                -2.69049815e-01,
                -4.44708621e-01,
                9.40019675e-04,
                -1.16329608e-02,
                4.10610505e-02,
                -7.21261779e-04,
                -4.48376547e-01,
                -2.51065061e-02,
                -1.63318890e-01,
                -2.20804520e-02,
                6.02709767e-04,
                -4.37222303e-01,
                2.77357148e-02,
                -2.26673576e-04,
                -2.19315980e-02,
                -8.09694817e-03,
                -4.34792885e-01,
                -2.77790456e-01,
                4.08487897e-05,
                -2.04880529e-03,
                6.86210300e-03,
                -5.37917969e-06,
                -2.77935611e-01,
                -2.27740480e-03,
                -2.52657567e-02,
                -3.12172687e-03,
                2.86572890e-05,
                -2.76213838e-01,
                4.90216098e-03,
                -3.54316106e-06,
                -3.12177332e-03,
                -7.76427590e-04,
                -2.76161148e-01,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                -0.24083366,
                -0.25062307,
                -0.03000751,
                0.03521329,
                -0.25062307,
                -0.34373788,
                -0.08330737,
                0.08333185,
                -0.03000751,
                -0.08330737,
                -0.00308714,
                -0.07338497,
                0.03521329,
                0.08333185,
                -0.07338497,
                -0.18934109,
            ]
        ),
    )
