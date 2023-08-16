import os
import pytest
from sys import platform

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


@pytest.mark.parametrize("with_cholesky", [True, False])
@pytest.mark.parametrize("with_scaling", [True, False])
def test_arm_reaching_torque_driven_implicit(with_cholesky, with_scaling):
    from bioptim.examples.stochastic_optimal_control import arm_reaching_torque_driven_implicit as ocp_module

    if platform == "win32":
        return

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
            np.testing.assert_almost_equal(f[0, 0], 54.28952476221059)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 54.2895173562166)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 7.405993991395128e-06)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.92561035, 1.29037204]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.41464303, -0.30980662]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.39128527, 0.2434043]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [0.00255935, 0.02549327, -0.00768962, 0.06599174, -0.15552605, -0.06097116, 0.05700404, -0.11164645]
                ),
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412559e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11116271e00,
                        1.69057247e-04,
                        -1.24156418e-02,
                        -2.33928682e-05,
                        2.01589626e-05,
                        1.11130210e00,
                        -1.18395058e-04,
                        -1.24145857e-02,
                        -4.64368985e-03,
                        -1.52153796e-02,
                        1.11740783e00,
                        2.10544735e-03,
                        -1.81416951e-03,
                        -1.71892973e-02,
                        1.06554819e-02,
                        1.11731267e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(cov[:, -2], np.zeros((16,)))

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000000e00,
                        -1.65451610e-13,
                        -1.00000000e-01,
                        -3.31608690e-13,
                        -1.62834877e-13,
                        1.00000000e00,
                        -6.45493156e-13,
                        -1.00000000e-01,
                        6.16740727e-04,
                        -2.21468046e-01,
                        1.03953533e00,
                        3.26199789e-02,
                        -7.58441133e-02,
                        4.71987094e-01,
                        7.31476378e-02,
                        9.94322787e-01,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        -4.25077873e-18,
                        -5.92422089e-17,
                        4.69316470e-17,
                        1.25991655e-16,
                        -1.86588834e-05,
                        -4.49579778e-05,
                        -4.94431763e-18,
                        -4.95063458e-17,
                        3.76688314e-17,
                        1.03008301e-16,
                        -1.69869006e-05,
                        -4.08985761e-05,
                        -1.35470069e00,
                        1.13061087e00,
                        -4.32731295e-02,
                        -7.55445151e-02,
                        -1.58053086e-05,
                        -3.80812013e-05,
                        1.13061087e00,
                        -3.80089856e00,
                        6.97384180e-02,
                        1.60904792e-01,
                        7.36570801e-02,
                        1.73483980e-01,
                    ]
                ),
            )
        else:
            # Check objective function value
            np.testing.assert_almost_equal(f[0, 0], 54.463880332948555)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 54.37236353986075)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0.09151679308782121)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906593, 2.24586775]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.92560997, 1.29037212]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.408751, -0.30095452]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.39684473, 0.25717582]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [-1.0390094, -0.32579281, -0.01435754, 0.29677779, -5.18715674, -1.27447891, 2.94219156, 0.40862461]
                ),
            )
            np.testing.assert_almost_equal(
                ref[:, 0], np.array([2.81907379e-02, 2.84412556e-01, -9.59019515e-08, 1.42444836e-08])
            )
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11173226e00,
                        3.62203081e-04,
                        -1.17455865e-02,
                        -8.99380103e-05,
                        -1.01817292e-05,
                        1.11145413e00,
                        -6.27317649e-05,
                        -1.24881122e-02,
                        -5.59028257e-02,
                        -3.25981162e-02,
                        1.05710275e00,
                        8.09457476e-03,
                        9.16617438e-04,
                        -3.08713589e-02,
                        5.64604205e-03,
                        1.12393041e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                cov[:, -2],
                np.array(
                    [
                        2.81275916e-11,
                        3.73897947e-12,
                        5.61418456e-14,
                        1.31046019e-11,
                        3.73897947e-12,
                        -3.26513891e-11,
                        -1.29612870e-10,
                        -5.10667200e-12,
                        5.61418456e-14,
                        -1.29612870e-10,
                        -4.85963481e-10,
                        -3.19895432e-11,
                        1.31046019e-11,
                        -5.10667200e-12,
                        -3.19895432e-11,
                        -4.26135270e-12,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000001e00,
                        1.18312230e-08,
                        -1.00000005e-01,
                        1.56604917e-08,
                        3.22425585e-09,
                        1.00000000e00,
                        5.97468411e-10,
                        -9.99999951e-02,
                        -1.46383198e-01,
                        -3.77984261e-01,
                        4.13434834e-01,
                        -7.43556380e-02,
                        -5.58600887e-02,
                        3.70574596e-01,
                        -1.78901369e-02,
                        9.90615974e-01,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        1.44001519e-13,
                        -1.82442019e-13,
                        -2.81356498e-12,
                        -2.19486039e-12,
                        3.73686590e-02,
                        3.40909428e-02,
                        2.12486222e-13,
                        -8.57326932e-14,
                        1.83965118e-12,
                        9.52685515e-13,
                        2.03653259e-02,
                        1.85722301e-02,
                        -1.35578553e00,
                        1.12296286e00,
                        -2.13784280e-01,
                        -1.48398296e-01,
                        2.86719675e-02,
                        2.61285946e-02,
                        1.12296286e00,
                        -3.78683550e00,
                        5.46604622e-02,
                        1.67249211e-01,
                        2.08364656e-01,
                        1.96334541e-01,
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
            np.testing.assert_almost_equal(f[0, 0], 54.28951360888063, decimal=4)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 54.289512025867886, decimal=4)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0, decimal=4)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.9256103, 1.29037205]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.4145378, -0.3098442]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.39137504, 0.24336774]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [0.00221146, 0.02623768, -0.00761546, 0.0652775, -0.15924041, -0.04456824, 0.04320116, -0.05492831]
                ),
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907791e-02, 2.84412560e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11116494e00,
                        1.68474280e-04,
                        -1.24145415e-02,
                        -1.73877094e-05,
                        1.38105810e-05,
                        1.11130403e00,
                        -1.21598227e-04,
                        -1.24315458e-02,
                        -4.84499568e-03,
                        -1.51626852e-02,
                        1.11730873e00,
                        1.56489535e-03,
                        -1.24295227e-03,
                        -1.73625871e-02,
                        1.09438404e-02,
                        1.11883912e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                cov[:, -2],
                np.array(
                    [
                        0.00063083,
                        0.00063075,
                        0.00063048,
                        0.00063075,
                        0.0006304,
                        0.00063048,
                        0.00063075,
                        0.0006304,
                        0.0006304,
                        0.00063048,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000000e00,
                        3.60343520e-16,
                        -1.00000000e-01,
                        6.74473442e-13,
                        9.17639615e-16,
                        1.00000000e00,
                        1.09913004e-15,
                        -1.00000000e-01,
                        -4.24361455e-03,
                        -2.17537642e-01,
                        1.03462378e00,
                        2.15310722e-02,
                        -6.36864518e-02,
                        4.61284037e-01,
                        8.57746684e-02,
                        1.02335384e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        2.89688836e-14,
                        -2.59753607e-14,
                        6.14097556e-14,
                        1.05872998e-13,
                        -6.30107285e-03,
                        3.60479386e-03,
                        -1.62877804e-14,
                        1.62295870e-14,
                        -3.58432044e-14,
                        -6.30330592e-14,
                        8.72761938e-04,
                        7.64959168e-03,
                        -1.35469952e00,
                        1.13061848e00,
                        -4.38221202e-02,
                        -7.44597142e-02,
                        7.53272577e-03,
                        8.64080976e-03,
                        1.13061848e00,
                        -3.80091245e00,
                        7.23035897e-02,
                        1.58769538e-01,
                        3.50414294e-02,
                        7.47132363e-02,
                    ]
                ),
            )
        else:
            # Check objective function value
            np.testing.assert_almost_equal(f[0, 0], 54.37668076964529)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 54.295453694426094)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0.0812270752191952)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.3490659, 2.24586771]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.92560995, 1.29037231]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.41102869, -0.31202548]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.39425576, 0.2432391]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [-1.02396298, -0.32267796, 0.13955518, 0.34290332, -4.85369264, -1.12807811, 2.60922174, 0.39782026]
                ),
            )
            np.testing.assert_almost_equal(
                ref[:, 0], np.array([2.81907555e-02, 2.84412569e-01, -7.77739045e-08, 1.19990756e-08])
            )
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11176079e00,
                        4.25887567e-04,
                        -1.17522321e-02,
                        -9.03563609e-05,
                        2.69353064e-06,
                        1.11146583e00,
                        -5.49871334e-05,
                        -1.24924773e-02,
                        -5.84713427e-02,
                        -3.83298872e-02,
                        1.05770083e00,
                        8.13208821e-03,
                        -2.42408697e-04,
                        -3.19246879e-02,
                        4.94885177e-03,
                        1.12432324e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                cov[:, -2],
                np.array(
                    [
                        0.00061489,
                        0.00056242,
                        0.00066602,
                        0.000584,
                        0.00061708,
                        0.00068358,
                        0.00064752,
                        0.00064571,
                        0.00060739,
                        0.00064715,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000000e00,
                        2.27762643e-10,
                        -9.99999993e-02,
                        2.33158028e-09,
                        8.15995773e-11,
                        1.00000000e00,
                        -1.87628470e-09,
                        -1.00000000e-01,
                        -2.18938571e-01,
                        -4.37925676e-01,
                        3.12368259e-01,
                        -1.12194069e-01,
                        -4.71621030e-02,
                        3.47777066e-01,
                        -6.41612919e-03,
                        1.02283471e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        -7.79408350e-13,
                        3.22715615e-12,
                        1.81654874e-12,
                        -5.83097502e-12,
                        -7.12605524e-05,
                        1.04268017e-03,
                        3.97505383e-12,
                        -1.03884279e-13,
                        3.79392908e-11,
                        1.91038777e-11,
                        1.67551189e-02,
                        1.43156257e-02,
                        -1.35458608e00,
                        1.13073561e00,
                        -1.94248154e-01,
                        -1.38165232e-01,
                        1.15214021e-04,
                        7.43859444e-05,
                        1.13073561e00,
                        -3.80101779e00,
                        7.40473200e-02,
                        1.68649827e-01,
                        1.23713659e-01,
                        9.71259408e-02,
                    ]
                ),
            )


def test_arm_reaching_torque_driven_collocations():
    from bioptim.examples.stochastic_optimal_control import arm_reaching_torque_driven_collocations as ocp_module

    return  # TODO: readd this test when the gug on the order of reorder_to_vector is fixed
    final_time = 0.4
    n_shooting = 4
    ee_final_position = np.array([9.359873986980460e-12, 0.527332023564034])

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
    np.testing.assert_almost_equal(f[0, 0], 7066.7926705880955)

    # detailed cost values
    np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 78.47204821578093)
    np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 6988.320622372315)
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
    np.testing.assert_almost_equal(q[:, 0], np.array([0.41128607, 2.36393904]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.91450978, 1.47825178]))
    np.testing.assert_almost_equal(qdot[:, 0], np.array([-0.01227987, -0.00442274]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([-0.01649727, -0.0137153]))

    np.testing.assert_almost_equal(tau[:, 0], np.array([0.28164146, -0.71694685]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-0.2789166, 0.83394751]))

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array(
            [-2.2311138, -7.88424867, 0.10891849, -2.50891714, -2.03501113, -9.14607336, -0.81082578, -3.96627336]
        ),
    )
    np.testing.assert_almost_equal(ref[:, 0], np.array([0.0102648, 0.12900788, 0.03473712, 0.01714053]))
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                -0.03780896,
                0.01428981,
                0.01329116,
                0.0136845,
                -0.13110697,
                0.01511638,
                0.01233682,
                0.01759883,
                -0.12401404,
                0.00574906,
                0.00545047,
                0.00770349,
                0.01409505,
                -0.03905632,
                0.01315758,
                0.01010664,
                0.0132242,
                -0.13156635,
                0.01041309,
                0.0114354,
                0.00522018,
                -0.12361953,
                0.00471038,
                0.00646891,
                0.01266273,
                0.01516653,
                -0.03945049,
                0.01487932,
                0.01021474,
                0.01886193,
                -0.13657921,
                0.01491626,
                0.004885,
                0.00727845,
                -0.12477003,
                0.00625501,
                0.002933,
                0.0052785,
                -0.00021098,
                -0.02575473,
                -0.03393887,
                -0.03565643,
                -0.02612223,
                -0.11649588,
                -0.00968069,
                -0.01105033,
                -0.0044019,
                -0.12087302,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                0.01362556,
                0.01064286,
                0.02551549,
                -0.03343807,
                0.01064286,
                0.00840897,
                0.01840482,
                -0.01834393,
                0.02551549,
                0.01840482,
                0.0291329,
                0.00457209,
                -0.03343807,
                -0.01834393,
                0.00457209,
                -0.12020213,
            ]
        ),
    )
