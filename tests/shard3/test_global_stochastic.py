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
    np.testing.assert_almost_equal(f[0, 0], 13.322871634584153)

    # detailed cost values
    np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 0.6783119392800068)
    np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0.4573562887022045)
    np.testing.assert_almost_equal(
        f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
    )

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (546, 1))

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
    np.testing.assert_almost_equal(q[:, -1], np.array([0.95993109, 1.15939485]))
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))
    np.testing.assert_almost_equal(
        mus_activations[:, 0], np.array([0.00559921, 0.00096835, 0.00175969, 0.01424529, 0.01341463,
       0.00648656])
    )
    np.testing.assert_almost_equal(
        mus_activations[:, -1], np.array([0.04856166, 0.09609582, 0.02063621, 0.0315381 , 0.00022286,
       0.0165601 ])
    )

    np.testing.assert_almost_equal(
        mus_excitations[:, 0], np.array([0.05453449, 0.07515539, 0.02860859, 0.01667135, 0.00352633,
       0.04392939])
    )
    np.testing.assert_almost_equal(
        mus_excitations[:, -2], np.array([0.05083793, 0.09576169, 0.02139706, 0.02832909, 0.00023962,
       0.02396517])
    )

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array(
            [
                0.00999995, 0.01, 0.00999999, 0.00999998, 0.00999997,
                0.00999999, 0.00999994, 0.01, 0.01, 0.00999998,
                0.00999997, 0.00999999, 0.0099997, 0.0099995, 0.00999953,
                0.00999958, 0.0099996, 0.00999953, 0.0099997, 0.0099995,
                0.00999953, 0.00999958, 0.0099996, 0.00999953,
            ]
        ),
    )
    np.testing.assert_almost_equal(ref[:, 0], np.array([0.00834655, 0.05367618, 0.00834655, 0.00834655]))
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                1.70810520e-01, 3.56453950e-03, 1.48227603e-02, 8.57938092e-03,
                -2.74736661e-02, 2.29699855e-02, 9.34744296e-02, -4.05561279e-02,
                1.03747046e-02, -2.69456243e-02, 9.24608816e-03, 1.56534006e-01,
                7.90121132e-03, 1.14023696e-02, 8.63061567e-02, -2.95050053e-02,
                -1.34736043e-02, 2.05592350e-02, 3.18864595e-02, -1.24806854e-02,
                -2.72650658e-02, 4.74437345e-02, 7.65728294e-02, 1.50545445e-02,
                -1.97257907e-01, 1.01220545e-01, 8.27850768e-01, -4.60172967e-01,
                6.85657953e-02, -3.64739879e-01, 1.05398530e-02, -7.63108417e-02,
                7.35733915e-03, 4.32844317e-02, 9.40540321e-01, -4.23529363e-01,
                -2.41629571e-01, 1.50980662e-01, 2.83683345e-01, -2.20090489e-01,
                8.98374479e-03, 8.00827704e-04, 7.53514379e-03, 5.98000313e-03,
                4.23095866e-02, 3.64376975e-02, 1.97804811e-02, 1.55818997e-03,
                -1.10621504e-02, 2.49629057e-02, 8.86397629e-03, -2.73081620e-03,
                7.93071078e-03, 8.57055714e-03, 1.07457907e-02, 1.04603417e-01,
                6.45608415e-03, 9.16055220e-03, 9.55375664e-03, 6.06502722e-03,
                9.77792061e-03, -3.57625997e-03, 4.94841001e-03, 7.38539951e-03,
                -4.36284627e-03, 1.23306909e-02, 7.64073642e-02, 2.58451398e-02,
                -1.19784814e-04, 2.79657076e-02, 8.40556268e-03, -5.06587091e-04,
                9.42249163e-03, 7.95998211e-03, -1.41585209e-02, 1.68244003e-02,
                2.95987301e-02, 9.51675252e-02, 4.83155620e-03, 3.01937740e-03,
                9.06928287e-03, -1.11586453e-03, 7.25722813e-03, 7.09660591e-03,
                -2.52062529e-02, 2.18948538e-02, 8.37855333e-03, 8.06247374e-03,
                9.69920902e-02, 1.89391527e-02, 8.39077342e-03, -1.48700041e-03,
                9.47333066e-03, 8.64491341e-03, 4.03005838e-03, 8.47777890e-03,
                2.53974474e-02, -1.64248894e-03, 1.02776900e-02, 9.74841774e-02,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                0.00033791, 0.00039624, 0.00070543, 0.00124988, 0.00021535,
                0.00029579, 0.00024912, 0.00028454, 0.00025029, 0.00030357,
                0.00039624, 0.00061519, 0.00019818, 0.00228786, 0.00029938,
                0.00042956, 0.00038645, 0.00039457, 0.00036173, 0.00042616,
                0.00070543, 0.00019818, 0.00482193, -0.00067968, 0.00027328,
                0.00027578, 0.00012372, 0.00035437, 0.00024831, 0.00035016,
                0.00124988, 0.00228786, -0.00067968, 0.01031238, 0.00110132,
                0.00158725, 0.00147344, 0.00143574, 0.00134504, 0.00155263,
                0.00021535, 0.00029938, 0.00027328, 0.00110132, 0.00015521,
                0.00021834, 0.00019183, 0.00020435, 0.00018451, 0.00021946,
                0.00029579, 0.00042956, 0.00027578, 0.00158725, 0.00021834,
                0.00031178, 0.00027831, 0.00028783, 0.00026257, 0.00031046,
                0.00024912, 0.00038645, 0.00012372, 0.00147344, 0.00019183,
                0.00027831, 0.00025442, 0.00025227, 0.00023393, 0.00027342,
                0.00028454, 0.00039457, 0.00035437, 0.00143574, 0.00020435,
                0.00028783, 0.00025227, 0.00026958, 0.00024298, 0.00028959,
                0.00025029, 0.00036173, 0.00024831, 0.00134504, 0.00018451,
                0.00026257, 0.00023393, 0.00024298, 0.00022139, 0.00026183,
                0.00030357, 0.00042616, 0.00035016, 0.00155263, 0.00021946,
                0.00031046, 0.00027342, 0.00028959, 0.00026183, 0.00031148,
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
def test_arm_reaching_torque_driven_implicit(with_cholesky):
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
    np.testing.assert_equal(g.shape, (338, 1))

    # Check some of the solution values
    states, controls, stochastic_variables = (
        sol.states,
        sol.controls,
        sol.stochastic_variables,
    )
    q, qdot = states["q"], states["qdot"]
    tau = controls["tau"]

    if not with_cholesky:
        # Check objective function value
        np.testing.assert_almost_equal(f[0, 0], 273.5607574226802)

        # detailed cost values
        np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 273.5544356233155)
        np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0.00632179936471477)
        np.testing.assert_almost_equal(
            f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
        )

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
        np.testing.assert_almost_equal(
            f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
        )

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
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
        np.testing.assert_almost_equal(qdot[:, -2], np.array([0, 0]))

        np.testing.assert_almost_equal(tau[:, 0], np.array([0.74341393, -0.38470965]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-0.69875678, 0.44426507]))

        np.testing.assert_almost_equal(
            k[:, 0],
            np.array(
                [0.01531877, 0.01126498, 0.01593056, 0.01857115, -0.00125035, -0.00515613, 0.00340021, -0.01075679]
            ),
        )
        np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907762e-02, 2.84412559e-01, 0, 0]))
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
                    0,
                    0,
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
                    0,
                    0,
                    0,
                    0,
                    -2.10526798e-04,
                    8.92312491e-04,
                    0,
                    0,
                    0,
                    0,
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
