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


def test_arm_reaching_torque_driven_implicit():
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
    )

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(4)
    solver.set_nlp_scaling_method("none")

    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 273.55627111505794)

    # detailed cost values
    np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 273.5562960834589)
    np.testing.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], -2.496840094602077e-05)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (338, 1))

    # Check some of the results
    states, controls, stochastic_variables = (
        sol.states,
        sol.controls,
        sol.stochastic_variables,
    )
    q, qdot = states["q"], states["qdot"]
    tau = controls["tau"]
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
    np.testing.assert_almost_equal(q[:, -2], np.array([0.92561256, 1.29036753]))
    np.testing.assert_almost_equal(qdot[:, 0], np.array([-2.82172788e-11, -1.23274702e-10]))
    np.testing.assert_almost_equal(qdot[:, -2], np.array([7.65668575e-10, 5.59702668e-10]))

    np.testing.assert_almost_equal(tau[:, 0], np.array([0.74344014, -0.38479402]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-0.6987582, 0.44431225]))

    np.testing.assert_almost_equal(
        k[:, 0],
        np.array([0.02004026, 0.0069783, 0.01389716, 0.01323812, -0.00142161, -0.00646875, 0.00394643, -0.01447342]),
    )
    np.testing.assert_almost_equal(
        ref[:, 0], np.array([2.81907783e-02, 2.84412560e-01, 2.86053354e-11, 3.23037015e-11])
    )
    np.testing.assert_almost_equal(
        m[:, 0],
        np.array(
            [
                1.11111457e00,
                3.88373567e-05,
                -1.24951503e-02,
                -7.04455780e-05,
                -1.40358119e-05,
                1.11184768e00,
                -2.10160021e-04,
                -1.23948958e-02,
                -3.10844387e-04,
                -3.49535274e-03,
                1.12456352e00,
                6.34010048e-03,
                1.26322734e-03,
                -6.62912775e-02,
                1.89144013e-02,
                1.11554061e00,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        cov[:, -2],
        np.array(
            [
                -1.78665942e-04,
                -1.51496351e-04,
                -1.21263671e-04,
                -4.05921692e-05,
                -1.51496351e-04,
                -1.26709767e-04,
                -9.61403341e-05,
                -3.41109001e-05,
                -1.21263671e-04,
                -9.61403341e-05,
                -7.78125309e-05,
                3.90777686e-06,
                -4.05921692e-05,
                -3.41109001e-05,
                3.90777686e-06,
                -5.29791556e-05,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        a[:, 3],
        np.array(
            [
                1.00000000e00,
                -1.22962760e-10,
                -1.00000000e-01,
                4.15084657e-12,
                -1.55692210e-10,
                1.00000000e00,
                -1.14130204e-10,
                -1.00000000e-01,
                3.94221859e-02,
                -4.10569380e-01,
                1.01639263e00,
                -1.87492281e-02,
                -9.52505814e-02,
                1.05939931e00,
                9.65948051e-03,
                1.08555264e00,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        c[:, 3],
        np.array(
            [
                1.72287273e-13,
                1.84947099e-13,
                1.07604338e-13,
                1.05774309e-13,
                1.51137629e-02,
                6.81012572e-02,
                1.00666354e-13,
                3.78518270e-14,
                7.34625949e-14,
                9.22662894e-14,
                1.22466921e-02,
                5.51958308e-02,
                -1.34786949e00,
                1.37816701e00,
                -1.77974876e-02,
                -3.20175900e-02,
                6.12337352e-03,
                2.72477423e-02,
                1.37816701e00,
                -4.28824687e00,
                2.28876944e-02,
                5.20590815e-02,
                7.43922133e-03,
                3.53387709e-02,
            ]
        ),
    )
