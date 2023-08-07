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
        mus_activations[:, 0], np.array([0.00559921, 0.00096835, 0.00175969, 0.01424529, 0.01341463, 0.00648656])
    )
    np.testing.assert_almost_equal(
        mus_activations[:, -1], np.array([0.04856166, 0.09609582, 0.02063621, 0.0315381, 0.00022286, 0.0165601])
    )

    np.testing.assert_almost_equal(
        mus_excitations[:, 0], np.array([0.05453449, 0.07515539, 0.02860859, 0.01667135, 0.00352633, 0.04392939])
    )
    np.testing.assert_almost_equal(
        mus_excitations[:, -2], np.array([0.05083793, 0.09576169, 0.02139706, 0.02832909, 0.00023962, 0.02396517])
    )

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array(
            [
                0.00999995,
                0.01,
                0.00999999,
                0.00999998,
                0.00999997,
                0.00999999,
                0.00999994,
                0.01,
                0.01,
                0.00999998,
                0.00999997,
                0.00999999,
                0.0099997,
                0.0099995,
                0.00999953,
                0.00999958,
                0.0099996,
                0.00999953,
                0.0099997,
                0.0099995,
                0.00999953,
                0.00999958,
                0.0099996,
                0.00999953,
            ]
        ),
    )
    np.testing.assert_almost_equal(ref[:, 0], np.array([0.00834655, 0.05367618, 0.00834655, 0.00834655]))
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                1.70810520e-01,
                3.56453950e-03,
                1.48227603e-02,
                8.57938092e-03,
                -2.74736661e-02,
                2.29699855e-02,
                9.34744296e-02,
                -4.05561279e-02,
                1.03747046e-02,
                -2.69456243e-02,
                9.24608816e-03,
                1.56534006e-01,
                7.90121132e-03,
                1.14023696e-02,
                8.63061567e-02,
                -2.95050053e-02,
                -1.34736043e-02,
                2.05592350e-02,
                3.18864595e-02,
                -1.24806854e-02,
                -2.72650658e-02,
                4.74437345e-02,
                7.65728294e-02,
                1.50545445e-02,
                -1.97257907e-01,
                1.01220545e-01,
                8.27850768e-01,
                -4.60172967e-01,
                6.85657953e-02,
                -3.64739879e-01,
                1.05398530e-02,
                -7.63108417e-02,
                7.35733915e-03,
                4.32844317e-02,
                9.40540321e-01,
                -4.23529363e-01,
                -2.41629571e-01,
                1.50980662e-01,
                2.83683345e-01,
                -2.20090489e-01,
                8.98374479e-03,
                8.00827704e-04,
                7.53514379e-03,
                5.98000313e-03,
                4.23095866e-02,
                3.64376975e-02,
                1.97804811e-02,
                1.55818997e-03,
                -1.10621504e-02,
                2.49629057e-02,
                8.86397629e-03,
                -2.73081620e-03,
                7.93071078e-03,
                8.57055714e-03,
                1.07457907e-02,
                1.04603417e-01,
                6.45608415e-03,
                9.16055220e-03,
                9.55375664e-03,
                6.06502722e-03,
                9.77792061e-03,
                -3.57625997e-03,
                4.94841001e-03,
                7.38539951e-03,
                -4.36284627e-03,
                1.23306909e-02,
                7.64073642e-02,
                2.58451398e-02,
                -1.19784814e-04,
                2.79657076e-02,
                8.40556268e-03,
                -5.06587091e-04,
                9.42249163e-03,
                7.95998211e-03,
                -1.41585209e-02,
                1.68244003e-02,
                2.95987301e-02,
                9.51675252e-02,
                4.83155620e-03,
                3.01937740e-03,
                9.06928287e-03,
                -1.11586453e-03,
                7.25722813e-03,
                7.09660591e-03,
                -2.52062529e-02,
                2.18948538e-02,
                8.37855333e-03,
                8.06247374e-03,
                9.69920902e-02,
                1.89391527e-02,
                8.39077342e-03,
                -1.48700041e-03,
                9.47333066e-03,
                8.64491341e-03,
                4.03005838e-03,
                8.47777890e-03,
                2.53974474e-02,
                -1.64248894e-03,
                1.02776900e-02,
                9.74841774e-02,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                0.00033791,
                0.00039624,
                0.00070543,
                0.00124988,
                0.00021535,
                0.00029579,
                0.00024912,
                0.00028454,
                0.00025029,
                0.00030357,
                0.00039624,
                0.00061519,
                0.00019818,
                0.00228786,
                0.00029938,
                0.00042956,
                0.00038645,
                0.00039457,
                0.00036173,
                0.00042616,
                0.00070543,
                0.00019818,
                0.00482193,
                -0.00067968,
                0.00027328,
                0.00027578,
                0.00012372,
                0.00035437,
                0.00024831,
                0.00035016,
                0.00124988,
                0.00228786,
                -0.00067968,
                0.01031238,
                0.00110132,
                0.00158725,
                0.00147344,
                0.00143574,
                0.00134504,
                0.00155263,
                0.00021535,
                0.00029938,
                0.00027328,
                0.00110132,
                0.00015521,
                0.00021834,
                0.00019183,
                0.00020435,
                0.00018451,
                0.00021946,
                0.00029579,
                0.00042956,
                0.00027578,
                0.00158725,
                0.00021834,
                0.00031178,
                0.00027831,
                0.00028783,
                0.00026257,
                0.00031046,
                0.00024912,
                0.00038645,
                0.00012372,
                0.00147344,
                0.00019183,
                0.00027831,
                0.00025442,
                0.00025227,
                0.00023393,
                0.00027342,
                0.00028454,
                0.00039457,
                0.00035437,
                0.00143574,
                0.00020435,
                0.00028783,
                0.00025227,
                0.00026958,
                0.00024298,
                0.00028959,
                0.00025029,
                0.00036173,
                0.00024831,
                0.00134504,
                0.00018451,
                0.00026257,
                0.00023393,
                0.00024298,
                0.00022139,
                0.00026183,
                0.00030357,
                0.00042616,
                0.00035016,
                0.00155263,
                0.00021946,
                0.00031046,
                0.00027342,
                0.00028959,
                0.00026183,
                0.00031148,
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
    )

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(4)
    solver.set_nlp_scaling_method("none")

    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.0009949559696219028)

    # detailed cost values
    np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 0.0009947791440719843)
    np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0)
    np.testing.assert_almost_equal(
        f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
    )

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (214, 1))

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
    np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.92481582, 1.29144455]))
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))
    np.testing.assert_almost_equal(qddot[:, 0], np.array([0, 0]))
    np.testing.assert_almost_equal(qddot[:, -1], np.array([0, 0]))

    np.testing.assert_almost_equal(qdddot[:, 0], np.array([38.88569235, -59.56207164]))
    np.testing.assert_almost_equal(qdddot[:, -2], np.array([32.98170709, -59.56049898]))

    np.testing.assert_almost_equal(tau[:, 0], np.array([1.32767652e-05, 1.32767652e-05]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([1.32767652e-05, 1.32767652e-05]))

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array(
            [
                4.53999320e-05,
                4.53999320e-05,
                5.29993608e-05,
                5.29993608e-05,
                1.46544478e-05,
                1.46544478e-05,
                1.46544550e-05,
                1.46544550e-05,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        ref[:, 0], np.array([2.81666286e-02, 2.84048250e-01, 1.32759716e-05, 1.32759716e-05])
    )
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                9.98685679e-01,
                1.32759716e-05,
                9.98805163e-02,
                1.32759716e-05,
                1.00000000e-02,
                1.32759716e-05,
                1.32759716e-05,
                9.98685679e-01,
                1.32759716e-05,
                9.98805163e-02,
                1.32759716e-05,
                1.00000000e-02,
                1.32759716e-05,
                1.32759716e-05,
                9.98685679e-01,
                1.32759716e-05,
                9.98805163e-02,
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                9.98685679e-01,
                1.32759716e-05,
                9.98805163e-02,
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                9.98685679e-01,
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                9.98685679e-01,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                9.94981789e-05,
                5.87995640e-09,
                7.45496761e-08,
                2.93126339e-09,
                8.22602351e-08,
                2.68422091e-09,
                5.87995640e-09,
                9.94981789e-05,
                2.93126339e-09,
                7.45496761e-08,
                2.68422091e-09,
                8.22602351e-08,
                7.45496761e-08,
                2.93126339e-09,
                2.58657288e-07,
                3.52419757e-11,
                3.97931483e-07,
                5.14120545e-11,
                2.93126339e-09,
                7.45496761e-08,
                3.52419757e-11,
                2.58657288e-07,
                5.14120545e-11,
                3.97931483e-07,
                8.22602351e-08,
                2.68422091e-09,
                3.97931483e-07,
                5.14120545e-11,
                9.94764852e-07,
                6.46744617e-11,
                2.68422091e-09,
                8.22602351e-08,
                5.14120545e-11,
                3.97931483e-07,
                6.46744617e-11,
                9.94764852e-07,
            ]
        ),
    )


# TODO: add test when scaling PR is merged
@pytest.mark.parametrize("with_cholesky", [True, False])
@pytest.mark.parametrize("with_scaling", [True, False])
def test_arm_reaching_torque_driven_implicit(with_cholesky, with_scaling):
    from bioptim.examples.stochastic_optimal_control import arm_reaching_torque_driven_implicit as ocp_module

    final_time = 0.8
    n_shooting = 4
    ee_final_position = np.array([9.359873986980460e-12, 0.527332023564034])

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
        with_cholesky=with_cholesky,
        with_scaling=with_scaling,
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
