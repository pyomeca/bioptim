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
                9.24608816e-03,
                -2.72650658e-02,
                1.05398530e-02,
                8.98374479e-03,
                8.86397629e-03,
                9.77792061e-03,
                8.40556268e-03,
                9.06928287e-03,
                8.39077342e-03,
                3.56453950e-03,
                1.56534006e-01,
                4.74437345e-02,
                -7.63108417e-02,
                8.00827704e-04,
                -2.73081620e-03,
                -3.57625997e-03,
                -5.06587091e-04,
                -1.11586453e-03,
                -1.48700041e-03,
                1.48227603e-02,
                7.90121132e-03,
                7.65728294e-02,
                7.35733915e-03,
                7.53514379e-03,
                7.93071078e-03,
                4.94841001e-03,
                9.42249163e-03,
                7.25722813e-03,
                9.47333066e-03,
                8.57938092e-03,
                1.14023696e-02,
                1.50545445e-02,
                4.32844317e-02,
                5.98000313e-03,
                8.57055714e-03,
                7.38539951e-03,
                7.95998211e-03,
                7.09660591e-03,
                8.64491341e-03,
                -2.74736661e-02,
                8.63061567e-02,
                -1.97257907e-01,
                9.40540321e-01,
                4.23095866e-02,
                1.07457907e-02,
                -4.36284627e-03,
                -1.41585209e-02,
                -2.52062529e-02,
                4.03005838e-03,
                2.29699855e-02,
                -2.95050053e-02,
                1.01220545e-01,
                -4.23529363e-01,
                3.64376975e-02,
                1.04603417e-01,
                1.23306909e-02,
                1.68244003e-02,
                2.18948538e-02,
                8.47777890e-03,
                9.34744296e-02,
                -1.34736043e-02,
                8.27850768e-01,
                -2.41629571e-01,
                1.97804811e-02,
                6.45608415e-03,
                7.64073642e-02,
                2.95987301e-02,
                8.37855333e-03,
                2.53974474e-02,
                -4.05561279e-02,
                2.05592350e-02,
                -4.60172967e-01,
                1.50980662e-01,
                1.55818997e-03,
                9.16055220e-03,
                2.58451398e-02,
                9.51675252e-02,
                8.06247374e-03,
                -1.64248894e-03,
                1.03747046e-02,
                3.18864595e-02,
                6.85657953e-02,
                2.83683345e-01,
                -1.10621504e-02,
                9.55375664e-03,
                -1.19784814e-04,
                4.83155620e-03,
                9.69920902e-02,
                1.02776900e-02,
                -2.69456243e-02,
                -1.24806854e-02,
                -3.64739879e-01,
                -2.20090489e-01,
                2.49629057e-02,
                6.06502722e-03,
                2.79657076e-02,
                3.01937740e-03,
                1.89391527e-02,
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
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                9.98685679e-01,
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                9.98805163e-02,
                1.32759716e-05,
                9.98685679e-01,
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                1.32759716e-05,
                9.98805163e-02,
                1.32759716e-05,
                9.98685679e-01,
                1.32759716e-05,
                1.32759716e-05,
                1.00000000e-02,
                1.32759716e-05,
                9.98805163e-02,
                1.32759716e-05,
                9.98685679e-01,
                1.32759716e-05,
                1.32759716e-05,
                1.00000000e-02,
                1.32759716e-05,
                9.98805163e-02,
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

    if platform != "linux":
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
            np.testing.assert_almost_equal(f[0, 0], 54.69985224379656, decimal=3)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 54.32008339136118, decimal=3)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0.37976885243538916, decimal=3)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906581, 2.2458677]), decimal=3)
            np.testing.assert_almost_equal(q[:, -1], np.array([0.92561119, 1.29037047]), decimal=3)
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.41236619, -0.3019402]), decimal=3)
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.39331919, 0.25051353]), decimal=3)

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [0.00357632, 0.02445081, -0.00986843, 0.06489354, -0.09927389, -0.05395245, 0.02192656, -0.11455955]
                ),
                decimal=3,
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907942e-02, 2.84412566e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11112755e00,
                        2.59583056e-05,
                        -1.58230391e-03,
                        -2.38966622e-03,
                        1.49765134e-04,
                        1.11130920e00,
                        -1.36566090e-02,
                        -1.77763801e-02,
                        -1.24383519e-02,
                        -1.12172481e-04,
                        1.11954774e00,
                        1.00935740e-02,
                        -2.64523768e-05,
                        -1.24142567e-02,
                        2.43275276e-03,
                        1.11728557e00,
                    ]
                ),
                decimal=3,
            )

            np.testing.assert_almost_equal(cov[:, -2], np.zeros((16,)))

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000000e00,
                        -8.52697691e-12,
                        6.55855871e-03,
                        -8.00703258e-02,
                        -8.52767701e-12,
                        1.00000000e00,
                        -2.18239828e-01,
                        4.70460344e-01,
                        -1.00000000e-01,
                        -5.25564645e-11,
                        1.04992986e00,
                        6.75331769e-02,
                        -2.27751861e-11,
                        -1.00000000e-01,
                        3.49290194e-02,
                        9.93850189e-01,
                    ]
                ),
                decimal=3,
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        -9.55596295e-17,
                        -1.54067811e-16,
                        -1.35545994e00,
                        1.12586820e00,
                        -2.44280819e-15,
                        -2.27063971e-15,
                        1.12586820e00,
                        -3.79227614e00,
                        2.35553605e-15,
                        2.12315614e-15,
                        -4.07317154e-02,
                        6.78173223e-02,
                        5.57501840e-15,
                        5.07550371e-15,
                        -7.50727482e-02,
                        1.61831876e-01,
                        4.29241800e-12,
                        4.18835826e-12,
                        9.54651163e-02,
                        9.82494768e-02,
                        2.65129632e-12,
                        2.44681382e-12,
                        -5.45976741e-02,
                        1.56733807e-01,
                    ]
                ),
                decimal=3,
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

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.41464357, -0.30982089]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.39128281, 0.24339136]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [-0.00096149, 0.02035493, -0.0088978, 0.0607317, -0.06245199, -0.02129132, -0.00063203, -0.03873214]
                ),
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412560e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11111524e00,
                        2.08790013e-05,
                        -3.71501575e-04,
                        -1.87911021e-03,
                        1.51566560e-04,
                        1.11130725e00,
                        -1.36409905e-02,
                        -1.76521629e-02,
                        -1.24519566e-02,
                        -1.18401839e-04,
                        1.12067610e00,
                        1.06561656e-02,
                        -3.18364285e-05,
                        -1.24290108e-02,
                        2.86527872e-03,
                        1.11861097e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                cov[:, -2],
                np.array(
                    [
                        0.00063053,
                        0.00063051,
                        0.00063053,
                        0.00063051,
                        0.00063051,
                        0.00063053,
                        0.0006309,
                        0.0006309,
                        0.0006309,
                        0.00063092,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000000e00,
                        -2.98544470e-13,
                        8.02353293e-03,
                        -7.40942144e-02,
                        -3.03310221e-13,
                        1.00000000e00,
                        -2.08526281e-01,
                        4.49404079e-01,
                        -1.00000000e-01,
                        1.60699818e-13,
                        1.05086197e00,
                        8.27928363e-02,
                        1.67273745e-13,
                        -1.00000000e-01,
                        2.37204061e-02,
                        1.03155176e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        1.17083508e-15,
                        -9.89169199e-16,
                        -1.35469932e00,
                        1.13061913e00,
                        -1.96731741e-15,
                        2.13883136e-15,
                        1.13061913e00,
                        -3.80091351e00,
                        3.09571194e-15,
                        -3.66065927e-15,
                        -3.30069508e-02,
                        8.97866654e-02,
                        5.95869682e-15,
                        -6.25543803e-15,
                        -6.76760308e-02,
                        1.58295490e-01,
                        -4.90430438e-13,
                        8.31960455e-14,
                        6.67559604e-02,
                        5.34086751e-02,
                        3.70399088e-13,
                        -3.06926255e-13,
                        -6.38435790e-03,
                        2.40412176e-02,
                    ]
                ),
            )
        else:
            # Check objective function value
            np.testing.assert_almost_equal(f[0, 0], 54.291122435161654)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 54.289504209637485)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0.0016182255241675885)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906586, 2.24586772]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.92561025, 1.29037213]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.41463662, -0.30982481]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.39128902, 0.24338939]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [-0.14874009, -0.07254171, 0.25687802, 0.38517585, 0.26043327, 0.11319856, -0.1197832, -0.13731224]
                ),
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907781e-02, 2.84412562e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11117800e00,
                        5.74496303e-06,
                        -6.02227021e-03,
                        -5.18291816e-04,
                        1.90782053e-04,
                        1.11146918e00,
                        -1.71716344e-02,
                        -3.22273064e-02,
                        -1.23773410e-02,
                        -8.05336127e-05,
                        1.11396106e00,
                        7.24817755e-03,
                        -7.90350142e-06,
                        -1.24521009e-02,
                        7.11391914e-04,
                        1.12068911e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                cov[:, -2],
                np.array(
                    [
                        0.00122792,
                        0.00120341,
                        0.00121168,
                        0.00118693,
                        0.0011951,
                        0.00121136,
                        0.00117985,
                        0.00118825,
                        0.00120402,
                        0.00122814,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        9.99999994e-01,
                        -4.18302471e-10,
                        -4.89315025e-02,
                        -6.22690957e-02,
                        -4.57777265e-10,
                        9.99999999e-01,
                        -2.50324579e-01,
                        3.41299838e-01,
                        -1.00000001e-01,
                        2.41176854e-09,
                        9.47809755e-01,
                        3.25135907e-02,
                        7.54210147e-11,
                        -9.99999999e-02,
                        -1.79887711e-02,
                        1.02668652e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        9.22250867e-12,
                        -2.88762428e-11,
                        -1.35469911e00,
                        1.13061978e00,
                        -1.53781787e-11,
                        3.60437290e-11,
                        1.13061978e00,
                        -3.80091459e00,
                        1.83034425e-11,
                        -3.41762066e-11,
                        -8.27123470e-03,
                        9.22737389e-02,
                        4.83015893e-11,
                        -1.20282422e-10,
                        -6.57808417e-02,
                        1.69050819e-01,
                        -7.18803280e-09,
                        3.56313048e-08,
                        8.15217455e-02,
                        4.28542941e-02,
                        3.27255319e-09,
                        -7.56135618e-09,
                        -8.27016953e-03,
                        2.96993918e-02,
                    ]
                ),
            )


def test_arm_reaching_torque_driven_collocations():
    from bioptim.examples.stochastic_optimal_control import arm_reaching_torque_driven_collocations as ocp_module

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
    np.testing.assert_almost_equal(f[0, 0], 427.1964048446773)

    # detailed cost values
    np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 432.1720966628036)
    np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], -4.975691818126331)
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
    np.testing.assert_almost_equal(q[:, -1], np.array([0.92528707, 1.29074965]))
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

    np.testing.assert_almost_equal(tau[:, 0], np.array([1.74670235, -1.02685616]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-1.66611244,  0.89557688]))

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array(
            [-0.0070663 ,  0.12872882, -0.04928026,  0.34387433, -0.0011865 , 0.08144921, -0.11150911,  0.14994164]
        ),
    )
    np.testing.assert_almost_equal(ref[:, 0], np.array([2.81676137e-02, 2.84063111e-01, 1.27344336e-05, 1.27344336e-05]))
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                9.99920715e-01, -4.11854315e-03, 2.90525360e-02, -1.04938345e-01,
                3.88306475e-03, 1.02971098e+00, 1.04514630e-01, 6.62884963e-01,
                9.78357140e-02, -4.57465514e-03, 9.41188170e-01, -1.20715456e-01,
                1.81733236e-03, 9.28715366e-02, 3.03216746e-02, 8.76735934e-01,
                -2.77702973e-01, 1.01862790e-03, -7.68818329e-03, 2.76801702e-02,
                -7.50021333e-04, -2.84233573e-01, -2.47787006e-02, -1.65603410e-01,
                -2.41041297e-02, 1.11904866e-03, -2.62138314e-01, 3.16680509e-02,
                -3.50468352e-04, -2.30524714e-02, -6.87981095e-03, -2.46538143e-01,
                -4.44077662e-01, 7.07737847e-04, -8.83700659e-03, 3.10407809e-02,
                -4.07608443e-05, -4.47306824e-01, -1.79003645e-02, -1.59574972e-01,
                -2.18732091e-02, 7.28451642e-04, -4.26675935e-01, 3.51042913e-02,
                -1.16592608e-04, -2.13658297e-02, -4.31646300e-03, -4.13351871e-01,
                -2.77422512e-01, 4.28889415e-05, -1.55118528e-03, 5.35116851e-03,
                1.03687508e-04, -2.77471537e-01, -1.56936263e-03, -2.46436788e-02,
                -3.11720317e-03, 2.88437370e-05, -2.74497551e-01, 6.02456151e-03,
                1.05073778e-05, -3.08780886e-03, -3.10808927e-04, -2.72693650e-01,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                0.04449698, -0.04720099, -0.00852083, 0.0292907, -0.04720099,
                -0.28603462, -0.03428146, 0.03041599, -0.00852083, -0.03428146,
                -0.01509507, -0.03243314, 0.0292907, 0.03041599, -0.03243314,
                -0.00564712,
            ]
        ),
    )
