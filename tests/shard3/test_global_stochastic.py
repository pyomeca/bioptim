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
    example_type = ExampleType.CIRCLE
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
        example_type=example_type,
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
    np.testing.assert_equal(g.shape, (378, 1))

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
            np.testing.assert_almost_equal(f[0, 0], 54.52503521402212)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 54.29902305061319)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"],  0.2260121634089394)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.92560991, 1.29037327]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.41458218, -0.30877801]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.3913934 ,  0.24451472]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [-0.08126391,  0.2555253 ,  0.14448397, -0.38543332,  0.02258506, -0.3548974 , -0.10106857,  0.21484998,
                     ]
                ),
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02,  2.84412560e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11118269e+00, -1.01088125e-04, -6.69274037e-03, 9.37614653e-03,
                        3.54328488e-04, 1.11077099e+00, -3.21976048e-02, 3.08588246e-02,
                        -1.23151867e-02, -4.31680585e-04, 1.10858202e+00, 3.85026923e-02,
                        -7.61182248e-05, -1.22808123e-02, 6.99433511e-03, 1.10499242e+00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(cov[:, -2], np.array([
                6.52543613e-05, 8.54789516e-05, -3.10706849e-05, 2.95591692e-04,
                8.54789516e-05, 7.09765598e-05, 5.20324750e-05, 5.09560540e-05,
                -3.10706849e-05, 5.20324750e-05, -4.73927717e-04, 1.46882408e-03,
                2.95591692e-04, 5.09560540e-05, 1.46882408e-03, -3.90068096e-03
            ]))

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000000e+00, 2.16573903e-26, -1.71680289e-01, 3.39804452e-01,
                        9.22703865e-27, 1.00000000e+00, -7.88173086e-01, 1.92833292e+00,
                        -1.00000000e-01, 2.66088779e-25, 6.83890524e-01, 9.76166402e-01,
                        7.85244297e-26, -1.00000000e-01, 1.46446396e-02, 1.03491985e+00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        2.10606720e-21, -2.76045022e-21, -1.35477674e+00, 1.12995089e+00,
                        5.61868682e-21, 1.03559080e-20, 1.12995089e+00, -3.79966503e+00,
                        2.10914685e-20, -1.81873389e-19, 2.60434644e-02, -9.18742242e-02,
                        1.53049468e-20, -2.77724515e-21, 6.86377099e-01, -1.83112718e+00,
                        -1.96374488e-22, -4.94581833e-24, 3.80735114e-01, -6.08404655e-01,
                        -9.71851671e-24, 1.15948045e-23, -2.68213921e-01, 6.79382306e-01,
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
            np.testing.assert_almost_equal(f[0, 0], 54.289516399688324)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 54.28951637226647)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.9256103 , 1.29037205]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.41464358, -0.30982087]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.39128281,  0.24339135]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [0.00034365,  0.01910308, -0.00163408,  0.04265227, -0.06063906, -0.02985536,  0.02308921, -0.09782673]
                ),
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412560e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11111940e+00, 8.60340848e-06, -7.46245214e-04, -7.74306837e-04,
                        1.55589438e-04, 1.11129681e+00, -1.40030495e-02, -1.67126449e-02,
                        -1.24516919e-02, -1.21015808e-04, 1.12065227e+00, 1.08914228e-02,
                        -1.43614542e-05, -1.24750189e-02, 1.29253101e-03, 1.12275170e+00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                cov[:, -2],
                np.array(
                    [
                        -3.29703946e-03, -6.92443684e-03, -5.37388642e-03, -6.92187179e-05,
                        -2.75968140e-03, 1.16584779e-03, 1.08350177e-02, 1.17285434e-02,
                        -3.08935653e-03, -4.05261428e-03,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000000e+00, -3.54249576e-13, -9.40356719e-04, -3.63835621e-02,
                        -2.92323448e-13, 1.00000000e+00, -2.17991954e-01, 4.79203390e-01,
                        -1.00000000e-01, 1.52414472e-13, 1.05656929e+00, 9.07298952e-02,
                        1.41930772e-13, -1.00000000e-01, 1.42863927e-02, 1.06655999e+00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        2.34516237e-16, 2.91978959e-16, -1.35469931e+00, 1.13061914e+00,
                        -1.50449070e-16, -2.53741932e-16, 1.13061914e+00, -3.80091353e+00,
                        5.35002433e-17, 1.58514475e-16, -1.87287883e-02, 5.18995316e-02,
                        3.67207110e-16, 5.57954047e-16, -4.21540750e-02, 9.41206379e-02,
                        -2.55674935e-13, -2.78423024e-13, 5.68101021e-02, 4.84875925e-02,
                        1.58634543e-14, 2.88777322e-14, -2.44511677e-03, 1.69785328e-02,
                    ]
                ),
            )
        else:
            # Check objective function value
            np.testing.assert_almost_equal(f[0, 0], 54.2895385839923)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 54.28951637525621)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 2.220873609222109e-05)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.9256103 , 1.29037205]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.41464358, -0.30982087]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.3912828 ,  0.24339133]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [0.30822601,  0.13163747,  0.18135185,  0.12970301, -0.33603948, -0.0991772 ,  0.14781477, -0.31955507]
                ),
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02,  2.84412560e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11107839e+00, 1.83578259e-06, 2.94500660e-03, -1.65220234e-04,
                        1.80438344e-04, 1.11137388e+00, -1.62394510e-02, -2.36495018e-02,
                        -1.25412756e-02, -1.16374741e-04, 1.12871480e+00, 1.04737280e-02,
                        8.08471140e-06, -1.26557497e-02, -7.27625392e-04, 1.13901747e+00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                cov[:, -2],
                np.array(
                    [
                        -9.24186729e-03, -3.01190231e-03, -1.15163152e-03, 7.35995233e-03,
                        2.43045884e-03, 2.69642882e-03, 9.30403706e-03, 9.10721054e-03,
                        2.82036728e-03, -4.39327505e-05,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000000e+00, -1.26787707e-13, -1.48740859e-02, -2.87793461e-02,
                        -3.67743992e-15, 1.00000000e+00, -2.18729452e-01, 3.93198331e-01,
                        -1.00000000e-01, -2.86587594e-15, 1.16150538e+00, 1.13413207e-01,
                        -2.13572267e-14, -1.00000000e-01, 2.30456912e-02, 1.18554985e+00
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        -7.03552104e-15, 7.44518746e-15, -1.35469931e+00, 1.13061914e+00,
                        2.64376120e-14, -2.30539451e-14, 1.13061914e+00, -3.80091354e+00,
                        -2.71324241e-14, 2.31863372e-14, -1.42800841e-02, 6.14052795e-02,
                        -4.01315780e-14, 3.61162915e-14, -4.18210813e-02, 9.12390290e-02,
                        -3.96519136e-12, 1.97499448e-12, 2.63915557e-02, 1.63879654e-03,
                        -1.44212826e-12, 1.16551814e-12, 2.21628743e-03, -6.69609165e-03,
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
    np.testing.assert_almost_equal(tau[:, -2], np.array([-1.66611244, 0.89557688]))

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array([-0.0070663, 0.12872882, -0.04928026, 0.34387433, -0.0011865, 0.08144921, -0.11150911, 0.14994164]),
    )
    np.testing.assert_almost_equal(
        ref[:, 0], np.array([2.81676137e-02, 2.84063111e-01, 1.27344336e-05, 1.27344336e-05])
    )
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                9.99920715e-01,
                -4.11854315e-03,
                2.90525360e-02,
                -1.04938345e-01,
                3.88306475e-03,
                1.02971098e00,
                1.04514630e-01,
                6.62884963e-01,
                9.78357140e-02,
                -4.57465514e-03,
                9.41188170e-01,
                -1.20715456e-01,
                1.81733236e-03,
                9.28715366e-02,
                3.03216746e-02,
                8.76735934e-01,
                -2.77702973e-01,
                1.01862790e-03,
                -7.68818329e-03,
                2.76801702e-02,
                -7.50021333e-04,
                -2.84233573e-01,
                -2.47787006e-02,
                -1.65603410e-01,
                -2.41041297e-02,
                1.11904866e-03,
                -2.62138314e-01,
                3.16680509e-02,
                -3.50468352e-04,
                -2.30524714e-02,
                -6.87981095e-03,
                -2.46538143e-01,
                -4.44077662e-01,
                7.07737847e-04,
                -8.83700659e-03,
                3.10407809e-02,
                -4.07608443e-05,
                -4.47306824e-01,
                -1.79003645e-02,
                -1.59574972e-01,
                -2.18732091e-02,
                7.28451642e-04,
                -4.26675935e-01,
                3.51042913e-02,
                -1.16592608e-04,
                -2.13658297e-02,
                -4.31646300e-03,
                -4.13351871e-01,
                -2.77422512e-01,
                4.28889415e-05,
                -1.55118528e-03,
                5.35116851e-03,
                1.03687508e-04,
                -2.77471537e-01,
                -1.56936263e-03,
                -2.46436788e-02,
                -3.11720317e-03,
                2.88437370e-05,
                -2.74497551e-01,
                6.02456151e-03,
                1.05073778e-05,
                -3.08780886e-03,
                -3.10808927e-04,
                -2.72693650e-01,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                0.04449698,
                -0.04720099,
                -0.00852083,
                0.0292907,
                -0.04720099,
                -0.28603462,
                -0.03428146,
                0.03041599,
                -0.00852083,
                -0.03428146,
                -0.01509507,
                -0.03243314,
                0.0292907,
                0.03041599,
                -0.03243314,
                -0.00564712,
            ]
        ),
    )
