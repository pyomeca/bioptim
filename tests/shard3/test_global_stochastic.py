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
    hand_final_position = np.array([9.359873986980460e-12, 0.527332023564034])
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
        hand_final_position=hand_final_position,
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
    hand_final_position = np.array([9.359873986980460e-12, 0.527332023564034])

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
    np.testing.assert_almost_equal(f[0, 0], 47.84234142370227)

    # detailed cost values
    np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 0.05981612198204287)
    np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 7.691441651074669)
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
    np.testing.assert_almost_equal(q[:, -1], np.array([0.927911, 1.27506452]))
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))
    np.testing.assert_almost_equal(qddot[:, 0], np.array([0, 0]))
    np.testing.assert_almost_equal(qddot[:, -1], np.array([0, 0]))

    np.testing.assert_almost_equal(qdddot[:, 0], np.array([0.00148095, 0.00148095]))
    np.testing.assert_almost_equal(qdddot[:, -2], np.array([0.00148095, 0.00148095]))

    np.testing.assert_almost_equal(tau[:, 0], np.array([0.35412622, -0.22800643]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-0.34741238, 0.23722038]))

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array(
            [
                0.0850057,
                0.93341438,
                0.07373058,
                0.38234598,
                0.00114773,
                0.00143368,
                0.00114589,
                0.00143052,
            ]
        ),
    )
    np.testing.assert_almost_equal(ref[:, 0], np.array([0.02549682, 0.24377349, 0.00148095, 0.00148095]))
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                7.83634893e-01,
                2.54187318e-01,
                -7.81857144e-01,
                2.44271921e00,
                1.10171073e-02,
                1.10171073e-02,
                1.71179604e-01,
                9.49679624e-01,
                1.64180651e00,
                9.07754598e-01,
                7.77661455e-03,
                7.77661455e-03,
                8.14450014e-02,
                -9.88728416e-03,
                8.07035277e-01,
                -1.06287579e-01,
                8.23859677e-04,
                8.23859677e-04,
                -2.01831916e-03,
                7.72321708e-02,
                -3.44807484e-02,
                7.58024151e-01,
                1.58861743e-03,
                1.58861743e-03,
                1.48094789e-03,
                1.48094789e-03,
                1.48094789e-03,
                1.48094789e-03,
                8.53386159e-01,
                1.48094789e-03,
                1.48094789e-03,
                1.48094789e-03,
                1.48094789e-03,
                1.48094789e-03,
                1.48094789e-03,
                8.53386159e-01,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                3.35959251e-02,
                -4.78514146e-02,
                1.32688111e-01,
                -1.92922727e-01,
                -5.44007932e-04,
                -5.44007932e-04,
                -4.78514146e-02,
                1.28135427e-01,
                -1.92350313e-01,
                5.14838845e-01,
                1.83905217e-03,
                1.83905217e-03,
                1.32688111e-01,
                -1.92350313e-01,
                7.24942742e-01,
                -1.04586706e00,
                -2.38182385e-03,
                -2.38182385e-03,
                -1.92922727e-01,
                5.14838845e-01,
                -1.04586706e00,
                2.80467284e00,
                7.86543559e-03,
                7.86543559e-03,
                -5.44007932e-04,
                1.83905217e-03,
                -2.38182385e-03,
                7.86543559e-03,
                2.85396821e-05,
                2.80129800e-05,
                -5.44007932e-04,
                1.83905217e-03,
                -2.38182385e-03,
                7.86543559e-03,
                2.80129800e-05,
                2.85396821e-05,
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
    hand_final_position = np.array([9.359873986980460e-12, 0.527332023564034])

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
        hand_final_position=hand_final_position,
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
            np.testing.assert_almost_equal(f[0, 0], 62.61240153041182)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 62.41253120355753)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0.19987032685429304)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.92560992, 1.29037324]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.42126532, -0.30424813]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.3933987, 0.36251051]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [-0.0770916, 0.24594264, 0.14356716, -0.37903073, 0.02556642, -0.33600195, -0.09768757, 0.21875505]
                ),
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412560e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11117933e00,
                        -9.89790360e-05,
                        -6.36575900e-03,
                        9.15679149e-03,
                        3.42798480e-04,
                        1.11078623e00,
                        -3.11148477e-02,
                        2.94613721e-02,
                        -1.22442829e-02,
                        -3.62937369e-04,
                        1.10217942e00,
                        3.23426016e-02,
                        -7.49239001e-05,
                        -1.20983199e-02,
                        6.87328702e-03,
                        1.08858017e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                cov[:, -2],
                np.array(
                    [
                        6.52095128e-05,
                        8.58550134e-05,
                        -3.43580868e-05,
                        3.04961560e-04,
                        8.58550134e-05,
                        7.51879842e-05,
                        3.90995716e-05,
                        8.48441793e-05,
                        -3.43580868e-05,
                        3.90995716e-05,
                        -4.29201762e-04,
                        1.31706534e-03,
                        3.04961560e-04,
                        8.48441793e-05,
                        1.31706534e-03,
                        -3.45141868e-03,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000000e00,
                        -8.11373860e-26,
                        -1.67781542e-01,
                        3.32131421e-01,
                        8.67187637e-27,
                        1.00000000e00,
                        -7.76947657e-01,
                        1.89631798e00,
                        -1.00000000e-01,
                        -9.63358420e-25,
                        6.71919094e-01,
                        8.86491256e-01,
                        -4.98418026e-25,
                        -1.00000000e-01,
                        4.44699545e-02,
                        8.69525177e-01,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        -3.27251991e-21,
                        -1.90706757e-20,
                        -1.35504801e00,
                        1.12904779e00,
                        -8.42798952e-21,
                        3.05675532e-20,
                        1.12904779e00,
                        -3.79816715e00,
                        8.03040833e-21,
                        2.53102785e-19,
                        3.01182416e-02,
                        -9.93007432e-02,
                        -6.05887964e-20,
                        1.18604326e-20,
                        6.81924202e-01,
                        -1.82105153e00,
                        -4.59863930e-23,
                        6.62189569e-24,
                        3.54626037e-01,
                        -5.62181724e-01,
                        8.68527931e-24,
                        2.81260929e-23,
                        -2.34792769e-01,
                        5.93239215e-01,
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
            np.testing.assert_almost_equal(f[0, 0], 62.40222244200586)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 62.40222242539446)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 1.6611394850611363e-08)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.9256103, 1.29037205]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.42135681, -0.30494449]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.39329963, 0.36152636]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [0.00227125, 0.01943845, -0.00045809, 0.04340353, -0.05890334, -0.02196787, 0.02044042, -0.08280278]
                ),
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412560e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11111643e00,
                        9.66024409e-06,
                        -4.78746311e-04,
                        -8.69421987e-04,
                        1.49883122e-04,
                        1.11128979e00,
                        -1.34894811e-02,
                        -1.60812429e-02,
                        -1.23773893e-02,
                        -6.07070546e-05,
                        1.11396504e00,
                        5.46363498e-03,
                        -2.04675057e-05,
                        -1.22691010e-02,
                        1.84207561e-03,
                        1.10421909e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                cov[:, -2],
                np.array(
                    [
                        0.00644836,
                        -0.00610657,
                        -0.00544246,
                        0.00168837,
                        0.0005854,
                        -0.00123564,
                        0.0103952,
                        0.01108306,
                        -0.00252879,
                        -0.00192049,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000000e00,
                        -2.17926087e-13,
                        1.26870284e-03,
                        -3.78607634e-02,
                        -2.56284891e-13,
                        1.00000000e00,
                        -2.19752019e-01,
                        4.81445536e-01,
                        -1.00000000e-01,
                        1.09505432e-13,
                        1.02554762e00,
                        4.87997817e-02,
                        9.63854391e-14,
                        -1.00000000e-01,
                        4.91622255e-02,
                        8.87744034e-01,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        2.24899604e-16,
                        4.19692812e-16,
                        -1.35499970e00,
                        1.12950726e00,
                        -1.16296826e-16,
                        -5.23075855e-16,
                        1.12950726e00,
                        -3.79903118e00,
                        6.93079055e-17,
                        4.43906938e-16,
                        -2.00791886e-02,
                        4.98852395e-02,
                        3.32248534e-16,
                        1.04710774e-15,
                        -4.28795369e-02,
                        9.36788627e-02,
                        -2.55942876e-13,
                        -2.73014494e-13,
                        5.33498922e-02,
                        4.09670671e-02,
                        5.18153700e-15,
                        3.81994693e-14,
                        -3.35841216e-04,
                        1.26309820e-02,
                    ]
                ),
            )
        else:
            # Check objective function value
            np.testing.assert_almost_equal(f[0, 0], 62.40224045726969)

            # detailed cost values
            np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 62.40222242578194)
            np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 1.8031487750452925e-05)
            np.testing.assert_almost_equal(
                f[0, 0], sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
            )

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.9256103, 1.29037205]))
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0]))

            np.testing.assert_almost_equal(tau[:, 0], np.array([0.42135677, -0.30494447]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-0.39329968, 0.3615263]))

            np.testing.assert_almost_equal(
                k[:, 0],
                np.array(
                    [0.31301854, 0.12182861, 0.19203473, 0.14751018, -0.32685328, -0.08325657, 0.14068481, -0.27395387]
                ),
            )
            np.testing.assert_almost_equal(ref[:, 0], np.array([2.81907786e-02, 2.84412560e-01, 0, 0]))
            np.testing.assert_almost_equal(
                m[:, 0],
                np.array(
                    [
                        1.11108266e00,
                        4.17488777e-06,
                        2.56083176e-03,
                        -3.75739664e-04,
                        1.81096926e-04,
                        1.11136793e00,
                        -1.62987234e-02,
                        -2.31135787e-02,
                        -1.24633259e-02,
                        -5.13628363e-05,
                        1.12169933e00,
                        4.62265629e-03,
                        -3.42511466e-06,
                        -1.24196246e-02,
                        3.08259114e-04,
                        1.11776621e00,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                cov[:, -2],
                np.array(
                    [
                        -1.07718751e-02,
                        -1.99371716e-03,
                        -5.91375606e-05,
                        7.26583106e-03,
                        1.90429781e-03,
                        1.99709199e-03,
                        1.06545354e-02,
                        9.52116195e-03,
                        1.98084162e-03,
                        8.98559860e-04,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                a[:, 3],
                np.array(
                    [
                        1.00000000e00,
                        -9.20399499e-14,
                        -1.61741543e-02,
                        -3.51768147e-02,
                        -6.60561379e-15,
                        1.00000000e00,
                        -2.23124489e-01,
                        3.93635442e-01,
                        -1.00000000e-01,
                        -1.09518183e-14,
                        1.13040430e00,
                        6.10450178e-02,
                        -2.95093488e-14,
                        -1.00000000e-01,
                        5.94097331e-02,
                        9.92723714e-01,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                c[:, 2],
                np.array(
                    [
                        -2.70484535e-14,
                        2.01986142e-14,
                        -1.35499970e00,
                        1.12950727e00,
                        6.51869829e-14,
                        -5.06432738e-14,
                        1.12950727e00,
                        -3.79903120e00,
                        -6.67237108e-14,
                        5.18030287e-14,
                        -1.31113529e-02,
                        6.16616592e-02,
                        -1.09243748e-13,
                        8.40241717e-14,
                        -4.16479142e-02,
                        9.08025965e-02,
                        1.60067573e-12,
                        -2.05218113e-13,
                        1.89372425e-02,
                        8.54507632e-03,
                        -2.53289376e-12,
                        1.87539717e-12,
                        1.30216832e-02,
                        -2.80276598e-02,
                    ]
                ),
            )


def test_arm_reaching_torque_driven_collocations():
    from bioptim.examples.stochastic_optimal_control import arm_reaching_torque_driven_collocations as ocp_module

    # if platform != "linux":
    #     return

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
