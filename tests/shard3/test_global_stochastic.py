import os
import pytest

import numpy as np
from casadi import DM, vertcat
from bioptim import Solver

from bioptim.examples.stochastic_optimal_control.arm_reaching_torque_driven_implicit import ExampleType


def test_arm_reaching_muscle_driven():
    from bioptim.examples.stochastic_optimal_control import arm_reaching_muscle_driven as ocp_module

    final_time = 0.8
    n_shooting = 4
    ee_final_position = np.array([9.359873986980460e-12, 0.527332023564034])
    problem_type = ExampleType.CIRCLE
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
        expand_dynamics=True,
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
    np.testing.assert_almost_equal(
        f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
    )

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
    problem_type = ExampleType.CIRCLE
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
        expand_dynamics=True,
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
    np.testing.assert_almost_equal(
        f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
    )

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


@pytest.mark.parametrize("with_cholesky", [True, False])
@pytest.mark.parametrize("with_scaling", [True, False])
def test_arm_reaching_torque_driven_implicit(with_cholesky, with_scaling):
    from bioptim.examples.stochastic_optimal_control import arm_reaching_torque_driven_implicit as ocp_module

    final_time = 0.8
    n_shooting = 4
    ee_final_position = np.array([9.359873986980460e-12, 0.527332023564034])
    problem_type = ExampleType.CIRCLE
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
        with_cholesky=with_cholesky,
        with_scaling=with_scaling,
        expand_dynamics=True,
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
    np.testing.assert_equal(g.shape, (298, 1))

    # Check some of the solution values
    states, controls, stochastic_variables = (
        sol.states,
        sol.controls,
        sol.stochastic_variables,
    )
    q, qdot = states["q"], states["qdot"]
    tau = controls["tau"]

    if not with_cholesky:
        # Check some of the results
        k, ref, m, cov, a, c = (
            stochastic_variables["k"],
            stochastic_variables["ref"],
            stochastic_variables["m"],
            stochastic_variables["cov"],
            stochastic_variables["a"],
            stochastic_variables["c"],
        )
        if not with_scaling:
            # Check objective function value
            np.testing.assert_almost_equal(f[0, 0], 273.056208256368)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 273.55443232300195)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], -0.4982240666339267)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
            np.testing.assert_almost_equal(q[:, -2], np.array([0.92561032, 1.29037201]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -2], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.74346464, -0.38483552]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.69868339, 0.4443794]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [0.01296001, 0.0151729, 0.02906942, 0.06261677, -0.00823412, 0.01481714, 0.00871672, -0.01520584]
                ),
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907785e-02, 2.84412560e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11118221e00,
                        3.71683348e-05,
                        -1.24373200e-02,
                        -2.16090372e-05,
                        7.37184159e-05,
                        1.11189426e00,
                        -1.43106814e-04,
                        -1.23522802e-02,
                        -6.39871990e-03,
                        -3.34514993e-03,
                        1.11935880e00,
                        1.94481300e-03,
                        -6.63465759e-03,
                        -7.04830550e-02,
                        1.28796127e-02,
                        1.11170522e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                cov[:, -2],
                np.array(
                    [
                        -0.01990132,
                        -0.01444097,
                        -0.00663085,
                        0.00533537,
                        -0.01444097,
                        -0.01047807,
                        -0.00480907,
                        0.0038745,
                        -0.00663085,
                        -0.00480907,
                        -0.00220108,
                        0.00178761,
                        0.00533537,
                        0.0038745,
                        0.00178761,
                        -0.00141832,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        9.99999999e-01,
                        -3.89248051e-10,
                        -1.00000000e-01,
                        5.60685086e-11,
                        -5.31189346e-10,
                        1.00000000e00,
                        -2.62632737e-10,
                        -9.99999999e-02,
                        1.28324889e-01,
                        -3.60278390e-01,
                        1.04245348e00,
                        -2.15111161e-04,
                        1.57193542e-02,
                        1.08534584e00,
                        8.89863990e-02,
                        1.11498775e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        -4.86512358e-16,
                        -1.64023010e-17,
                        1.55157922e-14,
                        5.50204479e-15,
                        -4.14092731e-03,
                        1.34757705e-02,
                        -3.01415705e-16,
                        2.26614661e-17,
                        1.32811267e-15,
                        1.55181903e-15,
                        -1.77237832e-03,
                        5.85831242e-03,
                        -1.33957032e00,
                        1.49127923e00,
                        -2.65655735e-01,
                        -1.30176087e-01,
                        -6.61835029e-03,
                        2.22924576e-02,
                        1.49127923e00,
                        -4.50503895e00,
                        -1.24194077e-01,
                        4.18323015e-01,
                        3.97942975e-03,
                        3.97939156e-02,
                    ]
                ),
            )
        else:
            # Check objective function value
            np.testing.assert_almost_equal(f[0, 0], 273.5628107530602)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 273.5544233523113)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0.008387400748937353)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
            np.testing.assert_almost_equal(q[:, -2], np.array([0.9256103, 1.29037204]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -2], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.74346544, -0.38486035]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.69871533, 0.44425582]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [-0.09791486, -0.0679742, -0.12074527, 0.1858609, -0.06137435, -0.03515881, 0.0348363, 0.01874756]
                ),
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412560e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11110861e00,
                        -4.84718987e-05,
                        -1.24603518e-02,
                        -3.97259623e-05,
                        -2.63818454e-05,
                        1.11193979e00,
                        -1.73663807e-04,
                        -1.23494311e-02,
                        2.25328303e-04,
                        4.36247279e-03,
                        1.12143166e00,
                        3.57533911e-03,
                        2.37438795e-03,
                        -7.45810894e-02,
                        1.56297646e-02,
                        1.11144881e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                cov[:, -2],
                np.array(
                    [
                        -9.89618931e-06,
                        -3.45028083e-06,
                        -5.23354601e-06,
                        -1.89764145e-06,
                        -3.45028083e-06,
                        -5.24965837e-07,
                        -1.16721665e-06,
                        3.48177426e-07,
                        -5.23354601e-06,
                        -1.16721665e-06,
                        -4.38132821e-06,
                        -1.44123754e-06,
                        -1.89764145e-06,
                        3.48177426e-07,
                        -1.44123754e-06,
                        1.47026674e-07,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000000e00,
                        1.61553458e-11,
                        -1.00000000e-01,
                        1.52883352e-12,
                        5.63303293e-11,
                        1.00000000e00,
                        4.42980195e-11,
                        -1.00000000e-01,
                        4.68554952e-02,
                        -3.95443327e-01,
                        9.63371458e-01,
                        -3.64541322e-02,
                        7.91555465e-03,
                        1.00177362e00,
                        9.35233799e-02,
                        1.10409036e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        9.28938457e-17,
                        5.15678168e-15,
                        3.44944333e-14,
                        -6.88950703e-14,
                        -2.50244799e-05,
                        1.55185240e-04,
                        -4.46548019e-16,
                        1.61949610e-15,
                        3.63754364e-15,
                        -1.36058508e-14,
                        -2.35730228e-06,
                        4.56783381e-05,
                        -1.33957038e00,
                        1.49128313e00,
                        -1.70735258e-01,
                        4.59339985e-02,
                        -8.01308070e-05,
                        5.60875899e-04,
                        1.49128313e00,
                        -4.50504681e00,
                        -9.24270965e-02,
                        5.28664179e-01,
                        -1.30927465e-03,
                        4.31811628e-02,
                    ]
                ),
            )
    else:
        # Check some of the results
        k, ref, m, cov, a, c = (
            stochastic_variables["k"],
            stochastic_variables["ref"],
            stochastic_variables["m"],
            stochastic_variables["cholesky_cov"],
            stochastic_variables["a"],
            stochastic_variables["c"],
        )
        if not with_scaling:
            # Check objective function value
            np.testing.assert_almost_equal(f[0, 0], 273.5544804584462)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 273.55441832245094)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 6.213599523135488e-05)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906586, 2.24586773]))
            np.testing.assert_almost_equal(q[:, -2], np.array([0.9256103, 1.29037205]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -2], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.74343734, -0.38484301]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.6987438, 0.44428394]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [0.01575478, 0.00959845, 0.0122835, 0.02700365, -0.00011816, -0.00313828, 0.00248586, -0.00900279]
                ),
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907790e-02, 2.84412560e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11111338e00,
                        3.21744605e-05,
                        -1.24953527e-02,
                        -6.87215761e-05,
                        -1.76630645e-05,
                        1.11186026e00,
                        -2.10812403e-04,
                        -1.24004709e-02,
                        -2.04345081e-04,
                        -2.89570145e-03,
                        1.12458174e00,
                        6.18494196e-03,
                        1.58967581e-03,
                        -6.74234364e-02,
                        1.89731163e-02,
                        1.11604238e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                cov[:, -2],
                np.array(
                    [
                        0.00012808,
                        0.00217046,
                        -0.00192125,
                        0.00029646,
                        -0.00189485,
                        0.00473035,
                        0.00028764,
                        -0.00187294,
                        0.00468956,
                        0.00063793,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000000e00,
                        -6.12890546e-15,
                        -1.00000000e-01,
                        2.00654184e-13,
                        3.52509182e-16,
                        1.00000000e00,
                        6.52020883e-15,
                        -1.00000000e-01,
                        3.66144396e-02,
                        -4.06364356e-01,
                        1.01302691e00,
                        -2.22411650e-02,
                        -7.99569554e-02,
                        1.05377677e00,
                        1.96259324e-02,
                        1.09513805e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        7.08833894e-14,
                        -1.13595953e-13,
                        3.46467243e-13,
                        7.54984304e-13,
                        2.89454086e-04,
                        -2.68862081e-04,
                        -2.30727726e-15,
                        8.87514306e-15,
                        -2.08030873e-14,
                        -4.91960928e-14,
                        -7.67159141e-04,
                        -1.14753770e-03,
                        -1.33957035e00,
                        1.49128148e00,
                        -8.34959562e-02,
                        -1.74278532e-01,
                        2.59554193e-03,
                        1.28134710e-02,
                        1.49128148e00,
                        -4.50504349e00,
                        1.74546823e-01,
                        4.05155067e-01,
                        1.22006763e-02,
                        3.91595961e-02,
                    ]
                ),
            )
        else:
            # Check objective function value
            np.testing.assert_almost_equal(f[0, 0], 273.5545072087765)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 273.55441832709585)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 8.888168070654883e-05)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
            np.testing.assert_almost_equal(q[:, -2], np.array([0.9256103, 1.29037205]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -2], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.74343703, -0.38484243]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.6987433, 0.4442823]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [0.03003774, 0.00019814, -0.00496659, 0.06786569, 0.00077753, -0.00276379, 0.00229205, -0.0089413]
                ),
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907790e-02, 2.84412560e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11111641e00,
                        2.20830949e-05,
                        -1.24957495e-02,
                        -6.85992228e-05,
                        -3.21760294e-05,
                        1.11188249e00,
                        -2.11030318e-04,
                        -1.24016010e-02,
                        -4.77215817e-04,
                        -1.98747854e-03,
                        1.12461745e00,
                        6.17393002e-03,
                        2.89584264e-03,
                        -6.94237411e-02,
                        1.89927286e-02,
                        1.11614409e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                cov[:, -2],
                np.array(
                    [
                        0.00012914,
                        0.00213648,
                        -0.00186397,
                        0.00029521,
                        -0.00180784,
                        0.00468509,
                        0.00028765,
                        -0.00178846,
                        0.00464194,
                        0.00063803,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000000e00,
                        -6.92562198e-15,
                        -1.00000000e-01,
                        2.48597578e-13,
                        1.41073005e-16,
                        1.00000000e00,
                        6.13116290e-15,
                        -1.00000000e-01,
                        3.59409944e-02,
                        -3.95536839e-01,
                        1.01322485e00,
                        -2.24593937e-02,
                        -7.91769055e-02,
                        1.02694882e00,
                        1.96713444e-02,
                        1.09595608e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        7.72818765e-14,
                        -1.24792518e-13,
                        3.99136307e-13,
                        8.97061408e-13,
                        3.78127571e-04,
                        -2.70116644e-04,
                        -1.42276414e-15,
                        8.08362525e-15,
                        -1.94267440e-14,
                        -4.66976386e-14,
                        -9.95904101e-04,
                        -1.03256027e-03,
                        -1.33957035e00,
                        1.49128152e00,
                        -8.62651596e-02,
                        -1.88433873e-01,
                        2.72610148e-03,
                        1.29914994e-02,
                        1.49128152e00,
                        -4.50504357e00,
                        1.88371616e-01,
                        4.40478866e-01,
                        1.33312912e-02,
                        3.87777856e-02,
                    ]
                ),
            )
