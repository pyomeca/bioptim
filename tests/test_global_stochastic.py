import os

import pytest
import numpy as np
from casadi import DM, vertcat
from bioptim import Solver

from .utils import TestUtils


def test_arm_reaching_muscle_driven():
    from bioptim.examples.stochastic_optimal_control import arm_reaching_muscle_driven as ocp_module

    final_time = 0.8
    n_shooting = 4
    ee_final_position = np.array([9.359873986980460e-12, 0.527332023564034])
    problem_type = "CIRCLE"
    force_field_magnitude = 0

    dt = 0.01
    motor_noise_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024
    motor_noise_magnitude = DM(np.array([motor_noise_std**2 / dt, motor_noise_std**2 / dt]))
    wPq_magnitude = DM(np.array([wPq_std**2 / dt, wPq_std**2 / dt]))
    wPqdot_magnitude = DM(np.array([wPqdot_std**2 / dt, wPqdot_std**2 / dt]))
    sensory_noise_magnitude = vertcat(wPq_magnitude, wPqdot_magnitude)

    ocp = ocp_module.prepare_socp(
        final_time=final_time,
        n_shooting=n_shooting,
        ee_final_position=ee_final_position,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        force_field_magnitude=force_field_magnitude,
        problem_type=problem_type,
    )

    # ocp.print(to_console=True, to_graph=False)  #TODO: check to adjust the print method

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(4)
    solver.set_nlp_scaling_method("none")

    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 8.741553107265094)

    # detailed cost values
    np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 0.4718449494109634)
    np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0.35077195834212055)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (426, 1))

    # Check some of the results
    states, controls, stochastic_variables, integrated_values = (
        sol.states,
        sol.controls,
        sol.stochastic_variables,
        sol.integrated_values,
    )
    q, qdot, mus_activations = states["q"], states["qdot"], states["muscles"]
    mus_excitations = controls["muscles"]
    k, ref, m = stochastic_variables["k"], stochastic_variables["ref"], stochastic_variables["m"]
    cov = integrated_values["cov"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
    np.testing.assert_almost_equal(q[:, -2], np.array([0.95993109, 1.15939485]))
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -2], np.array((0, 0)))
    np.testing.assert_almost_equal(
        mus_activations[:, 0], np.array([0.03325451, 0.032017, 0.01257345, 0.01548143, 0.00693818, 0.02362305])
    )
    np.testing.assert_almost_equal(
        mus_activations[:, -2], np.array([0.04064909, 0.07082433, 0.01877212, 0.02148688, 0.00063566, 0.02491778])
    )

    np.testing.assert_almost_equal(
        mus_excitations[:, 0], np.array([0.02751848, 0.06879444, 0.0450344, 0.00352429, 0.00024554, 0.03168447])
    )
    np.testing.assert_almost_equal(
        mus_excitations[:, -2], np.array([0.0181344, 0.04798581, 0.02832193, 0.01222864, 0.00102345, 0.0088184])
    )

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array(
            [
                0.00999994,
                0.01,
                0.00999999,
                0.00999998,
                0.00999997,
                0.00999999,
                0.00999993,
                0.01,
                0.00999999,
                0.00999997,
                0.00999996,
                0.00999999,
                0.00999956,
                0.00999936,
                0.0099994,
                0.00999943,
                0.00999947,
                0.00999938,
                0.00999956,
                0.00999936,
                0.0099994,
                0.00999943,
                0.00999947,
                0.00999938,
            ]
        ),
    )
    np.testing.assert_almost_equal(ref[:, 0], np.array([0.00812868, 0.05943125, 0.00812868, 0.00812868]))
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                1.89901532e-01,
                7.37629605e-03,
                1.47837967e-02,
                8.48588394e-03,
                -2.44306780e-02,
                2.51407475e-02,
                9.00788587e-02,
                -3.86962806e-02,
                1.00981570e-02,
                -2.30733399e-02,
                8.88238223e-03,
                1.87327235e-01,
                7.82078768e-03,
                1.15903691e-02,
                1.17688038e-01,
                -3.52754034e-02,
                -2.27927084e-02,
                2.51836130e-02,
                3.78113193e-02,
                -1.37819883e-02,
                -5.14407103e-02,
                3.27637737e-02,
                7.96690236e-02,
                1.55123192e-02,
                -2.61900313e-01,
                1.26166110e-01,
                8.45325058e-01,
                -4.65998670e-01,
                5.54005627e-02,
                -3.41642928e-01,
                9.68642180e-03,
                -3.90454599e-02,
                1.00389338e-02,
                4.65571712e-02,
                1.15928684e00,
                -4.77995399e-01,
                -2.83390614e-01,
                1.72800266e-01,
                3.32532185e-01,
                -2.48729412e-01,
                8.42008086e-03,
                6.40218692e-03,
                6.96861327e-03,
                5.54499059e-03,
                5.32573390e-02,
                4.00176588e-02,
                1.65010843e-02,
                2.94846560e-03,
                -1.30378055e-02,
                2.87952614e-02,
                8.50464084e-03,
                3.22234695e-03,
                7.45158123e-03,
                8.33625533e-03,
                2.18409550e-02,
                1.17673068e-01,
                -6.25692666e-04,
                1.26102981e-02,
                1.04194901e-02,
                8.77085875e-03,
                8.57092634e-03,
                1.50099719e-03,
                3.74691927e-03,
                6.92497563e-03,
                4.95548730e-03,
                1.26214602e-02,
                7.68623589e-02,
                3.22241750e-02,
                -6.67469147e-04,
                3.34241506e-02,
                8.42896447e-03,
                5.68106373e-03,
                9.34261990e-03,
                7.84752389e-03,
                -6.01895917e-03,
                1.73243552e-02,
                2.71792485e-02,
                1.09211191e-01,
                5.87081736e-03,
                3.98247822e-03,
                8.47609798e-03,
                4.46687085e-03,
                6.64618694e-03,
                6.79043263e-03,
                -1.92107784e-02,
                2.29710396e-02,
                2.10494165e-03,
                1.10678046e-02,
                1.09827984e-01,
                2.21189843e-02,
                8.45301225e-03,
                4.79815708e-03,
                9.35476131e-03,
                8.53257990e-03,
                1.45649552e-02,
                8.51068920e-03,
                2.16399421e-02,
                5.65570874e-05,
                1.16516931e-02,
                1.11793701e-01,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                2.97545513e-04,
                3.73674365e-04,
                5.65335837e-04,
                1.26296962e-03,
                1.78908823e-04,
                2.56302037e-04,
                2.04758638e-04,
                2.51139430e-04,
                2.13671861e-04,
                2.67740643e-04,
                3.73674365e-04,
                6.43521925e-04,
                2.20310986e-05,
                2.56475618e-03,
                2.77320345e-04,
                4.10989118e-04,
                3.60106616e-04,
                3.82950737e-04,
                3.44176352e-04,
                4.13060339e-04,
                5.65335837e-04,
                2.20310986e-05,
                4.76178176e-03,
                -1.26492852e-03,
                1.54594043e-04,
                1.38625113e-04,
                -2.92257216e-05,
                2.29920401e-04,
                1.17082151e-04,
                2.18245866e-04,
                1.26296962e-03,
                2.56475618e-03,
                -1.26492852e-03,
                1.25693491e-02,
                1.11684923e-03,
                1.64213060e-03,
                1.48910632e-03,
                1.51278060e-03,
                1.39205038e-03,
                1.63005958e-03,
                1.78908823e-04,
                2.77320345e-04,
                1.54594043e-04,
                1.11684923e-03,
                1.27124567e-04,
                1.85742168e-04,
                1.58722246e-04,
                1.75834398e-04,
                1.55680003e-04,
                1.88779851e-04,
                2.56302037e-04,
                4.10989118e-04,
                1.38625113e-04,
                1.64213060e-03,
                1.85742168e-04,
                2.75215982e-04,
                2.39677168e-04,
                2.56908116e-04,
                2.30103832e-04,
                2.77019648e-04,
                2.04758638e-04,
                3.60106616e-04,
                -2.92257216e-05,
                1.48910632e-03,
                1.58722246e-04,
                2.39677168e-04,
                2.17400680e-04,
                2.17787980e-04,
                2.00425615e-04,
                2.36415930e-04,
                2.51139430e-04,
                3.82950737e-04,
                2.29920401e-04,
                1.51278060e-03,
                1.75834398e-04,
                2.56908116e-04,
                2.17787980e-04,
                2.44108637e-04,
                2.14989931e-04,
                2.61993459e-04,
                2.13671861e-04,
                3.44176352e-04,
                1.17082151e-04,
                1.39205038e-03,
                1.55680003e-04,
                2.30103832e-04,
                2.00425615e-04,
                2.14989931e-04,
                1.92613893e-04,
                2.31674848e-04,
                2.67740643e-04,
                4.13060339e-04,
                2.18245866e-04,
                1.63005958e-03,
                1.88779851e-04,
                2.77019648e-04,
                2.36415930e-04,
                2.61993459e-04,
                2.31674848e-04,
                2.81571542e-04,
            ]
        ),
    )

    # simulate
    # TestUtils.simulate(sol)  # TODO: charbie -> fix this
    # for now, it does not match because the integration is done in the multinode_constraint


def test_arm_reaching_torque_driven_explicit():
    from bioptim.examples.stochastic_optimal_control import arm_reaching_torque_driven_explicit as ocp_module

    final_time = 0.8
    n_shooting = 4
    ee_final_position = np.array([9.359873986980460e-12, 0.527332023564034])
    problem_type = "CIRCLE"
    force_field_magnitude = 0

    dt = 0.01
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
        ee_final_position=ee_final_position,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        force_field_magnitude=force_field_magnitude,
        problem_type=problem_type,
    )

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(4)
    solver.set_nlp_scaling_method("none")

    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 561.0948164739301)

    # detailed cost values
    np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 0.0008000000000003249)
    np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 1.157506624022141e-06)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (177, 1))

    # Check some of the results
    states, controls, stochastic_variables, integrated_values = (
        sol.states,
        sol.controls,
        sol.stochastic_variables,
        sol.integrated_values,
    )
    q, qdot, qddot = states["q"], states["qdot"], states["qddot"]
    qdddot, tau = controls["qdddot"], controls["tau"]
    k, ref, m = stochastic_variables["k"], stochastic_variables["ref"], stochastic_variables["m"]
    cov = integrated_values["cov"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.34906532, 2.24586853]))
    np.testing.assert_almost_equal(q[:, -2], np.array([0.92562684, 1.29034226]))
    np.testing.assert_almost_equal(qdot[:, 0], np.array([-1.57899650e-07, 2.41254443e-07]))
    np.testing.assert_almost_equal(qdot[:, -2], np.array([-1.55837195e-07, 2.43938827e-07]))
    np.testing.assert_almost_equal(qddot[:, 0], np.array([-1.46193801e-08, 2.26087816e-08]))
    np.testing.assert_almost_equal(qddot[:, -2], np.array([1.44183405e-08, -2.27987398e-08]))

    np.testing.assert_almost_equal(qdddot[:, 0], np.array([288.28080732, -477.76320979]))
    np.testing.assert_almost_equal(qdddot[:, -2], np.array([288.28080727, -477.76320985]))

    np.testing.assert_almost_equal(tau[:, 0], np.array([-3.26925993e-09, 9.16460095e-09]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-4.74280539e-10, 6.73868548e-09]))

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array(
            [
                4.64142689e-03,
                4.64142689e-03,
                5.12302385e-03,
                5.12302385e-03,
                2.56803086e-05,
                2.56803086e-05,
                2.56805175e-05,
                2.56805175e-05,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        ref[:, 0], np.array([2.81907902e-02, 2.84412318e-01, 3.52327872e-09, -7.24617190e-08])
    )
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                9.99999999e-01,
                -5.52232186e-10,
                9.99999999e-02,
                -5.57651141e-11,
                9.99999999e-03,
                -6.57964810e-12,
                -5.52264505e-10,
                1.00000000e00,
                -5.57686152e-11,
                1.00000000e-01,
                -6.58097541e-12,
                1.00000000e-02,
                -4.41781681e-10,
                -2.20857111e-10,
                1.00000000e00,
                -2.23035836e-11,
                1.00000000e-01,
                -2.62121430e-12,
                -2.20831950e-10,
                -1.09915662e-10,
                -2.23011389e-11,
                1.00000000e00,
                -2.62129367e-12,
                1.00000000e-01,
                -8.82781541e-11,
                -4.41283053e-11,
                -8.81407350e-12,
                -4.46497667e-12,
                1.00000000e00,
                -5.30124698e-13,
                -4.40763326e-11,
                -2.19521899e-11,
                -4.45974851e-12,
                -2.12087407e-12,
                -5.29581943e-13,
                1.00000000e00,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                1.00022400e-04,
                -2.29971564e-13,
                7.19999318e-08,
                -3.40720475e-14,
                7.99999888e-08,
                -5.56777152e-15,
                -2.29971564e-13,
                1.00022400e-04,
                -3.40758584e-14,
                7.19999830e-08,
                -5.57519308e-15,
                7.99999972e-08,
                7.19999318e-08,
                -3.40758584e-14,
                2.60000000e-07,
                -2.93958506e-17,
                4.00000000e-07,
                -1.62077043e-17,
                -3.40720475e-14,
                7.19999830e-08,
                -2.93958506e-17,
                2.60000000e-07,
                -1.62165579e-17,
                4.00000000e-07,
                7.99999888e-08,
                -5.57519308e-15,
                4.00000000e-07,
                -1.62165579e-17,
                1.00000000e-06,
                -3.78659436e-18,
                -5.56777152e-15,
                7.99999972e-08,
                -1.62077043e-17,
                4.00000000e-07,
                -3.78659436e-18,
                1.00000000e-06,
            ]
        ),
    )


@pytest.mark.parametrize("cholesky_flag", [True, False])
def test_arm_reaching_torque_driven_implicit(cholesky_flag):
    from bioptim.examples.stochastic_optimal_control import arm_reaching_torque_driven_implicit as ocp_module

    final_time = 0.8
    n_shooting = 4
    ee_final_position = np.array([9.359873986980460e-12, 0.527332023564034])
    problem_type = "CIRCLE"
    force_field_magnitude = 0

    dt = 0.01
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
        ee_final_position=ee_final_position,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        force_field_magnitude=force_field_magnitude,
        problem_type=problem_type,
        cholesky_flag=cholesky_flag,
    )

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(4)
    solver.set_nlp_scaling_method("none")

    sol = ocp.solve(solver)

    # Check objective
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints values
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (338, 1))

    # Check some of the solution values
    states, controls, stochastic_variables = (
        sol.states,
        sol.controls,
        sol.stochastic_variables,
    )
    q, qdot = states["q"], states["qdot"]
    tau = controls["tau"]

    if not cholesky_flag:
        # Check objective function value
        np.testing.assert_almost_equal(f[0, 0], 273.5607574226802)

        # detailed cost values
        np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 273.5544356233155)
        np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0.00632179936471477)

        # Check some of the results
        k, ref, m, cov, a, c = (
            stochastic_variables["k"],
            stochastic_variables["ref"],
            stochastic_variables["m"],
            stochastic_variables["cov"],
            stochastic_variables["a"],
            stochastic_variables["c"],
        )

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
        np.testing.assert_almost_equal(q[:, -2], np.array([0.92561033, 1.29037199]))
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.17396838e-10, 3.20866997e-11]))
        np.testing.assert_almost_equal(qdot[:, -2], np.array([9.34909789e-11, 2.11489648e-10]))

        np.testing.assert_almost_equal(tau[:, 0], np.array([0.74345001, -0.38482294]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-0.69873141, 0.44427599]))

        np.testing.assert_almost_equal(
            k[:, 0],
            np.array(
                [0.01523928, 0.01556081, 0.03375243, 0.05246741, -0.00879659, 0.01632912, 0.00877083, -0.01418607]
            ),
        )
        np.testing.assert_almost_equal(
            ref[:, 0], np.array([2.81907783e-02, 2.84412560e-01, -3.84350362e-11, -6.31154841e-12])
        )
        np.testing.assert_almost_equal(
            m[:, 0],
            np.array(
                [
                    1.11118843e00,
                    4.33671754e-05,
                    -1.24355084e-02,
                    -1.92738667e-05,
                    7.71134318e-05,
                    1.11188416e00,
                    -1.41446562e-04,
                    -1.23501869e-02,
                    -6.95870145e-03,
                    -3.90308717e-03,
                    1.11919572e00,
                    1.73466734e-03,
                    -6.94022101e-03,
                    -6.95743720e-02,
                    1.27302256e-02,
                    1.11151678e00,
                ]
            ),
        )

        np.testing.assert_almost_equal(
            cov[:, -2],
            np.array(
                [
                    -8.80346012e-05,
                    -4.69527095e-05,
                    8.35293213e-05,
                    1.56300610e-04,
                    -4.69527095e-05,
                    -3.44615160e-05,
                    7.29566569e-05,
                    1.35527530e-04,
                    8.35293213e-05,
                    7.29566569e-05,
                    -2.26287713e-04,
                    -2.80104699e-04,
                    1.56300610e-04,
                    1.35527530e-04,
                    -2.80104699e-04,
                    -4.80293202e-04,
                ]
            ),
        )

        np.testing.assert_almost_equal(
            a[:, 3],
            np.array(
                [
                    9.99999997e-01,
                    -2.94678167e-09,
                    -1.00000003e-01,
                    -1.10867716e-10,
                    -2.79029383e-09,
                    9.99999997e-01,
                    -2.20141327e-09,
                    -1.00000000e-01,
                    3.02420237e-02,
                    -4.20998617e-01,
                    9.87060279e-01,
                    -3.08564981e-02,
                    -7.57260229e-03,
                    1.08904346e00,
                    9.53610677e-02,
                    1.11920251e00,
                ]
            ),
        )

        np.testing.assert_almost_equal(
            c[:, 3],
            np.array(
                [
                    -1.00026960e-12,
                    7.48271175e-12,
                    7.22298606e-12,
                    -1.12880911e-11,
                    1.11223927e-02,
                    -5.94119908e-03,
                    -4.41155433e-13,
                    4.22693132e-12,
                    4.55705335e-12,
                    -6.72512449e-12,
                    1.91518725e-02,
                    -9.67304622e-03,
                    -1.34051958e00,
                    1.51907793e00,
                    -4.45148469e-02,
                    1.50301525e-02,
                    -8.76853509e-02,
                    4.11969104e-02,
                    1.51907793e00,
                    -4.56171523e00,
                    -1.48051682e-02,
                    6.70065631e-02,
                    -9.60790421e-02,
                    4.58470601e-02,
                ]
            ),
        )
    else:
        # Check objective function value
        np.testing.assert_almost_equal(f[0, 0], 273.55612770421396)

        # detailed cost values
        np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 273.5560267942219)
        np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0.00010090999197482785)

        # Check some of the results
        k, ref, m, cov, a, c = (
            stochastic_variables["k"],
            stochastic_variables["ref"],
            stochastic_variables["m"],
            stochastic_variables["cholesky_cov"],
            stochastic_variables["a"],
            stochastic_variables["c"],
        )

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.34906586, 2.24586773]))
        np.testing.assert_almost_equal(q[:, -2], np.array([0.92561225, 1.29036811]))
        np.testing.assert_almost_equal(qdot[:, 0], np.array([5.99750888e-10, 6.36777753e-11]))
        np.testing.assert_almost_equal(qdot[:, -2], np.array([1.14034274e-09, 1.77348396e-09]))

        np.testing.assert_almost_equal(tau[:, 0], np.array([0.74341393, -0.38470965]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-0.69875678, 0.44426507]))

        np.testing.assert_almost_equal(
            k[:, 0],
            np.array(
                [0.01531877, 0.01126498, 0.01593056, 0.01857115, -0.00125035, -0.00515613, 0.00340021, -0.01075679]
            ),
        )
        np.testing.assert_almost_equal(
            ref[:, 0], np.array([2.81907762e-02, 2.84412559e-01, -1.82246478e-10, -3.02336569e-12])
        )
        np.testing.assert_almost_equal(
            m[:, 0],
            np.array(
                [
                    1.11111399e00,
                    3.60727553e-05,
                    -1.24942749e-02,
                    -6.89880004e-05,
                    -1.55956208e-05,
                    1.11185104e00,
                    -2.11345774e-04,
                    -1.23982943e-02,
                    -2.58769034e-04,
                    -3.24653915e-03,
                    1.12448474e00,
                    6.20892944e-03,
                    1.40361018e-03,
                    -6.65935312e-02,
                    1.90211252e-02,
                    1.11584654e00,
                ]
            ),
        )

        np.testing.assert_almost_equal(
            cov[:, -2],
            np.array(
                [
                    -4.46821105e-03,
                    -1.71731520e-03,
                    -1.02009010e-02,
                    -3.58196407e-03,
                    -6.50385303e-03,
                    9.57036181e-03,
                    2.93606642e-03,
                    -1.82590044e-04,
                    8.51698871e-03,
                    9.33034990e-05,
                ]
            ),
        )

        np.testing.assert_almost_equal(
            a[:, 3],
            np.array(
                [
                    1.00000000e00,
                    1.08524580e-09,
                    -9.99999991e-02,
                    2.72912724e-10,
                    -1.29617696e-10,
                    9.99999995e-01,
                    -5.48136491e-09,
                    -1.00000001e-01,
                    3.98959553e-02,
                    -4.10112704e-01,
                    1.01332373e00,
                    -2.12383714e-02,
                    -7.85600590e-02,
                    1.06875322e00,
                    2.12659225e-02,
                    1.09408356e00,
                ]
            ),
        )

        np.testing.assert_almost_equal(
            c[:, 3],
            np.array(
                [
                    -9.28531424e-12,
                    2.10560432e-11,
                    -8.74791141e-12,
                    -1.84391377e-11,
                    -2.10526798e-04,
                    8.92312491e-04,
                    2.61170664e-11,
                    -4.05508057e-11,
                    2.86516265e-11,
                    4.55817345e-11,
                    -2.73807889e-02,
                    6.59798851e-02,
                    -1.34796595e00,
                    1.37762629e00,
                    -2.57583298e-02,
                    -3.30498283e-02,
                    -2.29074656e-02,
                    5.59309999e-02,
                    1.37762629e00,
                    -4.28727508e00,
                    1.39574001e-02,
                    4.84712227e-02,
                    6.11340782e-04,
                    -4.45068909e-03,
                ]
            ),
        )
