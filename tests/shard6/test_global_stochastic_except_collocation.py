import platform

import numpy as np
import numpy.testing as npt
import pytest
from casadi import DM, vertcat

from bioptim import Solver, SolutionMerge, SolutionIntegrator
from ..utils import TestUtils

# TODO: All the tests in this file can be removed once the Leuven version of stochastic is removed from Bioptim


# Integrated values should be handled another way
# In the meantime, let's skip this test
# Please note that the test is very sensitive, so approximate values are enough
# @pytest.mark.parametrize("use_sx", [True, False])
# def test_arm_reaching_muscle_driven(use_sx):
#     from bioptim.examples.stochastic_optimal_control import arm_reaching_muscle_driven as ocp_module
#
#     final_time = 0.8
#     n_shooting = 4
#     hand_final_position = np.array([9.359873986980460e-12, 0.527332023564034])
#     example_type = ExampleType.CIRCLE
#     force_field_magnitude = 0
#
#     dt = 0.01
#     motor_noise_std = 0.05
#     wPq_std = 3e-4
#     wPqdot_std = 0.0024
#     motor_noise_magnitude = DM(np.array([motor_noise_std**2 / dt, motor_noise_std**2 / dt]))
#     wPq_magnitude = DM(np.array([wPq_std**2 / dt, wPq_std**2 / dt]))
#     wPqdot_magnitude = DM(np.array([wPqdot_std**2 / dt, wPqdot_std**2 / dt]))
#     sensory_noise_magnitude = vertcat(wPq_magnitude, wPqdot_magnitude)
#
#     ocp = ocp_module.prepare_socp(
#         final_time=final_time,
#         n_shooting=n_shooting,
#         hand_final_position=hand_final_position,
#         motor_noise_magnitude=motor_noise_magnitude,
#         sensory_noise_magnitude=sensory_noise_magnitude,
#         force_field_magnitude=force_field_magnitude,
#         example_type=example_type,
#         use_sx=use_sx,
#     )
#
#     # ocp.print(to_console=True, to_graph=False)  #TODO: check to adjust the print method
#
#     # Solver parameters
#     solver = Solver.IPOPT()
#     solver.set_maximum_iterations(4)
#     solver.set_nlp_scaling_method("none")
#
#     sol = ocp.solve(solver)
#
#     # Check objective function value
#     TestUtils.assert_objective_value(sol=sol, expected_value=13.32287163458417)
#
#     # detailed cost values
#     npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 0.6783119392800087)
#     npt.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0.4573562887022004)
#
#     # Check constraints
#     g = np.array(sol.constraints)
#     npt.assert_equal(g.shape, (546, 1))
#
#     # Check some of the results
#     states = sol.decision_states(to_merge=SolutionMerge.NODES)
#     controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
#     algebraic_states = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)
#
#     q, qdot, mus_activations = states["q"], states["qdot"], states["muscles"]
#     mus_excitations = controls["muscles"]
#     k, ref, m = algebraic_states["k"], algebraic_states["ref"], algebraic_states["m"]
#     # cov = integrated_values["cov"]
#
#     # initial and final position
#     npt.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
#     npt.assert_almost_equal(q[:, -1], np.array([0.95993109, 1.15939485]))
#     npt.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
#     npt.assert_almost_equal(qdot[:, -1], np.array((0, 0)))
#     npt.assert_almost_equal(
#         mus_activations[:, 0], np.array([0.00559921, 0.00096835, 0.00175969, 0.01424529, 0.01341463, 0.00648656])
#     )
#     npt.assert_almost_equal(
#         mus_activations[:, -1], np.array([0.04856166, 0.09609582, 0.02063621, 0.0315381, 0.00022286, 0.0165601])
#     )
#
#     npt.assert_almost_equal(
#         mus_excitations[:, 0], np.array([0.05453449, 0.07515539, 0.02860859, 0.01667135, 0.00352633, 0.04392939])
#     )
#     npt.assert_almost_equal(
#         mus_excitations[:, -2], np.array([0.05083793, 0.09576169, 0.02139706, 0.02832909, 0.00023962, 0.02396517])
#     )
#
#     npt.assert_almost_equal(
#         k[:, 0],
#         np.array(
#             [
#                 0.00999995,
#                 0.01,
#                 0.00999999,
#                 0.00999998,
#                 0.00999997,
#                 0.00999999,
#                 0.00999994,
#                 0.01,
#                 0.01,
#                 0.00999998,
#                 0.00999997,
#                 0.00999999,
#                 0.0099997,
#                 0.0099995,
#                 0.00999953,
#                 0.00999958,
#                 0.0099996,
#                 0.00999953,
#                 0.0099997,
#                 0.0099995,
#                 0.00999953,
#                 0.00999958,
#                 0.0099996,
#                 0.00999953,
#             ]
#         ),
#     )
#     npt.assert_almost_equal(ref[:, 0], np.array([0.00834655, 0.05367618, 0.00834655, 0.00834655]))
#     npt.assert_almost_equal(
#         m[:, 0],
#         np.array(
#             [
#                 1.70810520e-01,
#                 9.24608816e-03,
#                 -2.72650658e-02,
#                 1.05398530e-02,
#                 8.98374479e-03,
#                 8.86397629e-03,
#                 9.77792061e-03,
#                 8.40556268e-03,
#                 9.06928287e-03,
#                 8.39077342e-03,
#                 3.56453950e-03,
#                 1.56534006e-01,
#                 4.74437345e-02,
#                 -7.63108417e-02,
#                 8.00827704e-04,
#                 -2.73081620e-03,
#                 -3.57625997e-03,
#                 -5.06587091e-04,
#                 -1.11586453e-03,
#                 -1.48700041e-03,
#                 1.48227603e-02,
#                 7.90121132e-03,
#                 7.65728294e-02,
#                 7.35733915e-03,
#                 7.53514379e-03,
#                 7.93071078e-03,
#                 4.94841001e-03,
#                 9.42249163e-03,
#                 7.25722813e-03,
#                 9.47333066e-03,
#                 8.57938092e-03,
#                 1.14023696e-02,
#                 1.50545445e-02,
#                 4.32844317e-02,
#                 5.98000313e-03,
#                 8.57055714e-03,
#                 7.38539951e-03,
#                 7.95998211e-03,
#                 7.09660591e-03,
#                 8.64491341e-03,
#                 -2.74736661e-02,
#                 8.63061567e-02,
#                 -1.97257907e-01,
#                 9.40540321e-01,
#                 4.23095866e-02,
#                 1.07457907e-02,
#                 -4.36284627e-03,
#                 -1.41585209e-02,
#                 -2.52062529e-02,
#                 4.03005838e-03,
#                 2.29699855e-02,
#                 -2.95050053e-02,
#                 1.01220545e-01,
#                 -4.23529363e-01,
#                 3.64376975e-02,
#                 1.04603417e-01,
#                 1.23306909e-02,
#                 1.68244003e-02,
#                 2.18948538e-02,
#                 8.47777890e-03,
#                 9.34744296e-02,
#                 -1.34736043e-02,
#                 8.27850768e-01,
#                 -2.41629571e-01,
#                 1.97804811e-02,
#                 6.45608415e-03,
#                 7.64073642e-02,
#                 2.95987301e-02,
#                 8.37855333e-03,
#                 2.53974474e-02,
#                 -4.05561279e-02,
#                 2.05592350e-02,
#                 -4.60172967e-01,
#                 1.50980662e-01,
#                 1.55818997e-03,
#                 9.16055220e-03,
#                 2.58451398e-02,
#                 9.51675252e-02,
#                 8.06247374e-03,
#                 -1.64248894e-03,
#                 1.03747046e-02,
#                 3.18864595e-02,
#                 6.85657953e-02,
#                 2.83683345e-01,
#                 -1.10621504e-02,
#                 9.55375664e-03,
#                 -1.19784814e-04,
#                 4.83155620e-03,
#                 9.69920902e-02,
#                 1.02776900e-02,
#                 -2.69456243e-02,
#                 -1.24806854e-02,
#                 -3.64739879e-01,
#                 -2.20090489e-01,
#                 2.49629057e-02,
#                 6.06502722e-03,
#                 2.79657076e-02,
#                 3.01937740e-03,
#                 1.89391527e-02,
#                 9.74841774e-02,
#             ]
#         ),
#     )
#
#
# @pytest.mark.parametrize("use_sx", [True, False])
# def test_arm_reaching_torque_driven_explicit(use_sx):
#     from bioptim.examples.stochastic_optimal_control import arm_reaching_torque_driven_explicit as ocp_module
#
#     final_time = 0.8
#     n_shooting = 4
#     hand_final_position = np.array([9.359873986980460e-12, 0.527332023564034])
#
#     dt = 0.01
#     motor_noise_std = 0.05
#     wPq_std = 3e-4
#     wPqdot_std = 0.0024
#     motor_noise_magnitude = DM(np.array([motor_noise_std**2 / dt, motor_noise_std**2 / dt]))
#     wPq_magnitude = DM(np.array([wPq_std**2 / dt, wPq_std**2 / dt]))
#     wPqdot_magnitude = DM(np.array([wPqdot_std**2 / dt, wPqdot_std**2 / dt]))
#     sensory_noise_magnitude = vertcat(wPq_magnitude, wPqdot_magnitude)
#
#     bioptim_folder = TestUtils.bioptim_folder()
#
#     if use_sx:
#         with pytest.raises(
#             NotImplementedError, match="Wrong number or type of arguments for overloaded function 'MX_set'"
#         ):
#             ocp = ocp_module.prepare_socp(
#                 biorbd_model_path=bioptim_folder + "/examples/models/LeuvenArmModel.bioMod",
#                 final_time=final_time,
#                 n_shooting=n_shooting,
#                 hand_final_position=hand_final_position,
#                 motor_noise_magnitude=motor_noise_magnitude,
#                 sensory_noise_magnitude=sensory_noise_magnitude,
#                 use_sx=use_sx,
#             )
#         return
#
#     ocp = ocp_module.prepare_socp(
#         biorbd_model_path=bioptim_folder + "/examples/models/LeuvenArmModel.bioMod",
#         final_time=final_time,
#         n_shooting=n_shooting,
#         hand_final_position=hand_final_position,
#         motor_noise_magnitude=motor_noise_magnitude,
#         sensory_noise_magnitude=sensory_noise_magnitude,
#         use_sx=use_sx,
#     )
#
#     # Solver parameters
#     solver = Solver.IPOPT()
#     solver.set_maximum_iterations(4)
#     solver.set_nlp_scaling_method("none")
#
#     sol = ocp.solve(solver)
#
#     # Check objective function value
#     TestUtils.assert_objective_value(sol=sol, expected_value=46.99030175091475)
#
#     # detailed cost values
#     npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 0.055578630313992475)
#     npt.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 6.038226210163837)
#
#     # Check constraints
#     g = np.array(sol.constraints)
#     npt.assert_equal(g.shape, (214, 1))
#
#     # Check some of the results
#     states = sol.decision_states(to_merge=SolutionMerge.NODES)
#     controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
#     algebraic_states = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)
#
#     q, qdot, qddot = states["q"], states["qdot"], states["qddot"]
#     qdddot, tau = controls["qdddot"], controls["tau"]
#     k, ref, m = algebraic_states["k"], algebraic_states["ref"], algebraic_states["m"]
#     ocp.nlp[0].integrated_values["cov"].cx
#
#     # TODO Integrated value is not a proper way to go, it should be removed and recomputed at will
#     # cov = integrated_values["cov"]
#
#     # initial and final position
#     npt.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
#     npt.assert_almost_equal(q[:, -1], np.array([0.92702265, 1.27828413]))
#     npt.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
#     npt.assert_almost_equal(qdot[:, -1], np.array([0, 0]))
#     npt.assert_almost_equal(qddot[:, 0], np.array([0, 0]))
#     npt.assert_almost_equal(qddot[:, -1], np.array([0, 0]))
#
#     npt.assert_almost_equal(qdddot[:, 0], np.array([0.00124365, 0.00124365]))
#     npt.assert_almost_equal(qdddot[:, -2], np.array([0.00124365, 0.00124365]))
#
#     npt.assert_almost_equal(tau[:, 0], np.array([0.36186712, -0.2368119]))
#     npt.assert_almost_equal(tau[:, -2], np.array([-0.35709778, 0.18867995]))
#
#     npt.assert_almost_equal(
#         k[:, 0],
#         np.array(
#             [
#                 0.13824554,
#                 0.54172046,
#                 0.05570321,
#                 0.25169273,
#                 0.00095407,
#                 0.00121309,
#                 0.00095146,
#                 0.00121091,
#             ]
#         ),
#     )
#     npt.assert_almost_equal(ref[:, 0], np.array([0.02592847, 0.25028511, 0.00124365, 0.00124365]))
#     npt.assert_almost_equal(
#         m[:, 0],
#         np.array(
#             [
#                 8.36639386e-01,
#                 1.14636589e-01,
#                 -4.32594485e-01,
#                 1.10372277e00,
#                 4.73812392e-03,
#                 4.73812392e-03,
#                 8.01515210e-02,
#                 9.66785674e-01,
#                 7.40822199e-01,
#                 8.50818498e-01,
#                 6.74366790e-03,
#                 6.74366790e-03,
#                 7.92700393e-02,
#                 -8.94683551e-03,
#                 7.86796476e-01,
#                 -9.53722725e-02,
#                 6.55990825e-04,
#                 6.55990825e-04,
#                 -8.94995258e-04,
#                 7.69438075e-02,
#                 -2.33336654e-02,
#                 7.55054362e-01,
#                 1.59819032e-03,
#                 1.59819032e-03,
#                 1.24365477e-03,
#                 1.24365477e-03,
#                 1.24365477e-03,
#                 1.24365477e-03,
#                 8.76878178e-01,
#                 1.24365477e-03,
#                 1.24365477e-03,
#                 1.24365477e-03,
#                 1.24365477e-03,
#                 1.24365477e-03,
#                 1.24365477e-03,
#                 8.76878178e-01,
#             ]
#         ),
#     )
#
#
# @pytest.mark.parametrize("with_scaling", [True, False])
# @pytest.mark.parametrize("use_sx", [True, False])
# def test_arm_reaching_torque_driven_implicit(with_scaling, use_sx):
#     """
#     The values of this test is sketchy (when the controls can have different ControlTypes, the number of constraints should increase by 16.
#     """
#     from bioptim.examples.stochastic_optimal_control import arm_reaching_torque_driven_implicit as ocp_module
#
#     if not with_scaling and not use_sx:
#         pytest.skip("Redundant test")
#
#     final_time = 0.8
#     n_shooting = 4
#     hand_final_position = np.array([9.359873986980460e-12, 0.527332023564034])
#
#     dt = 0.01
#     motor_noise_std = 0.05
#     wPq_std = 3e-4
#     wPqdot_std = 0.0024
#     motor_noise_magnitude = DM(np.array([motor_noise_std**2 / dt, motor_noise_std**2 / dt]))
#     wPq_magnitude = DM(np.array([wPq_std**2 / dt, wPq_std**2 / dt]))
#     wPqdot_magnitude = DM(np.array([wPqdot_std**2 / dt, wPqdot_std**2 / dt]))
#     sensory_noise_magnitude = vertcat(wPq_magnitude, wPqdot_magnitude)
#
#     bioptim_folder = TestUtils.bioptim_folder()
#
#     ocp = ocp_module.prepare_socp(
#         biorbd_model_path=bioptim_folder + "/examples/models/LeuvenArmModel.bioMod",
#         final_time=final_time,
#         n_shooting=n_shooting,
#         hand_final_position=hand_final_position,
#         motor_noise_magnitude=motor_noise_magnitude,
#         sensory_noise_magnitude=sensory_noise_magnitude,
#         with_cholesky=False,
#         with_scaling=with_scaling,
#         use_sx=use_sx,
#     )
#
#     # Solver parameters
#     solver = Solver.IPOPT()
#     solver.set_maximum_iterations(4)
#     solver.set_nlp_scaling_method("none")
#
#     # Check the values which will be sent to the solver
#     np.random.seed(42)
#     expected = [2.262656e02, 6.982128e10, 2.695483e03] if with_scaling else [226.265556, 1064.856593, 377.623933]
#
#     TestUtils.compare_ocp_to_solve(
#         ocp,
#         v=np.random.rand(457, 1),
#         expected_v_f_g=expected,
#         decimal=6,
#     )
#     if platform.system() == "Windows":
#         return
#
#     sol = ocp.solve(solver)
#
#     # Check objective
#     f = np.array(sol.cost)
#     npt.assert_equal(f.shape, (1, 1))
#
#     # Check constraints values
#     g = np.array(sol.constraints)
#     npt.assert_equal(g.shape, (362, 1))
#
#     # Check some of the solution values
#     states = sol.decision_states(to_merge=SolutionMerge.NODES)
#     controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
#     algebraic_states = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)
#
#     q, qdot = states["q"], states["qdot"]
#     tau = controls["tau"]
#
#     # Check some of the results
#     k, ref, m, cov, a, c = (
#         controls["k"],
#         controls["ref"],
#         algebraic_states["m"],
#         controls["cov"],
#         controls["a"],
#         controls["c"],
#     )
#     if not with_scaling:
#         # Check objective function value
#         TestUtils.assert_objective_value(sol=sol, expected_value=95.49928267855638, decimal=4)
#
#         # detailed cost values
#         npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 95.43029410036674, decimal=4)
#         npt.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0.06898857818965713, decimal=4)
#
#         # initial and final position
#         npt.assert_almost_equal(q[:, 0], np.array([0.34906585, 2.24586773]))
#         npt.assert_almost_equal(q[:, -1], np.array([0.9255255, 1.290118]))
#         npt.assert_almost_equal(qdot[:, 0], np.array([0, 0]))
#         npt.assert_almost_equal(qdot[:, -1], np.array([0, 0]))
#
#         npt.assert_almost_equal(tau[:, 0], np.array([0.4160923, -0.2730973]))
#         npt.assert_almost_equal(tau[:, -2], np.array([-0.4087628, 0.3192567]))
#
#         npt.assert_almost_equal(ref[:, 0], np.array([2.8105910e-02, 2.8313229e-01, 4.6654784e-05, 4.6654784e-05]))
