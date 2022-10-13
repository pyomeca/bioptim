# import platform
# import os
#
# import numpy as np
# from bioptim import OdeSolver
#
# from .utils import TestUtils
#
#
# def test_xia_fatigable_muscles():
#     from bioptim.examples.fatigue import static_arm_with_fatigue as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#
#     model_path = f"{bioptim_folder}/models/arm26_constant.bioMod"
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=model_path,
#         final_time=0.9,
#         n_shooting=5,
#         fatigue_type="xia",
#         ode_solver=OdeSolver.COLLOCATION(),
#         torque_level=1,
#     )
#     sol = ocp.solve()
#
#     # Check objective function value
#     f = np.array(sol.cost)
#     np.testing.assert_equal(f.shape, (1, 1))
#     np.testing.assert_almost_equal(f[0, 0], 19.770521758810368)
#
#     # Check constraints
#     g = np.array(sol.constraints)
#     np.testing.assert_equal(g.shape, (552, 1))
#     np.testing.assert_almost_equal(g, np.zeros((552, 1)))
#
#     # Check some of the results
#     states, controls = sol.states, sol.controls
#     q, qdot, ma, mr, mf = states["q"], states["qdot"], states["muscles_ma"], states["muscles_mr"], states["muscles_mf"]
#     tau, muscles = controls["tau"], controls["muscles"]
#
#     # initial and final position
#     np.testing.assert_almost_equal(q[:, 0], np.array((0.07, 1.4)))
#     np.testing.assert_almost_equal(q[:, -1], np.array((1.64470726, 2.25033212)))
#
#     # initial and final velocities
#     np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(qdot[:, -1], np.array((-2.93853331, 3.00564551)))
#
#     # fatigue parameters
#     np.testing.assert_almost_equal(ma[:, 0], np.array((0, 0, 0, 0, 0, 0)))
#     np.testing.assert_almost_equal(
#         ma[:, -1], np.array((0.00739128, 0.00563555, 0.00159309, 0.02418655, 0.02418655, 0.00041913))
#     )
#     np.testing.assert_almost_equal(mr[:, 0], np.array((1, 1, 1, 1, 1, 1)))
#     np.testing.assert_almost_equal(
#         mr[:, -1], np.array((0.99260018, 0.99281414, 0.99707397, 0.97566527, 0.97566527, 0.99904065))
#     )
#     np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
#     np.testing.assert_almost_equal(
#         mf[:, -1],
#         np.array((8.54868154e-06, 1.55030599e-03, 1.33293886e-03, 1.48176210e-04, 1.48176210e-04, 5.40217808e-04)),
#     )
#
#     # initial and final controls
#     np.testing.assert_almost_equal(tau[:, 0], np.array((0.80920008, 1.66855572)))
#     np.testing.assert_almost_equal(tau[:, -2], np.array((0.81847388, -0.85234628)))
#
#     np.testing.assert_almost_equal(
#         muscles[:, 0],
#         np.array((6.22395441e-08, 4.38966513e-01, 3.80781292e-01, 2.80532297e-07, 2.80532297e-07, 2.26601989e-01)),
#     )
#     np.testing.assert_almost_equal(
#         muscles[:, -2],
#         np.array((8.86069119e-03, 1.17337666e-08, 1.28715148e-08, 2.02340603e-02, 2.02340603e-02, 2.16517945e-088)),
#     )
#
#     # save and load
#     TestUtils.save_and_load(sol, ocp, True)
#
#     # simulate
#     TestUtils.simulate(sol)
#
#
# def test_xia_stabilized_fatigable_muscles():
#     from bioptim.examples.fatigue import static_arm_with_fatigue as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#
#     model_path = f"{bioptim_folder}/models/arm26_constant.bioMod"
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=model_path,
#         final_time=0.9,
#         n_shooting=5,
#         fatigue_type="xia_stabilized",
#         ode_solver=OdeSolver.COLLOCATION(),
#         torque_level=1,
#     )
#     sol = ocp.solve()
#
#     # Check objective function value
#     f = np.array(sol.cost)
#     np.testing.assert_equal(f.shape, (1, 1))
#     np.testing.assert_almost_equal(f[0, 0], 19.770521758810393)
#
#     # Check constraints
#     g = np.array(sol.constraints)
#     np.testing.assert_equal(g.shape, (552, 1))
#     np.testing.assert_almost_equal(g, np.zeros((552, 1)))
#
#     # Check some of the results
#     states, controls = sol.states, sol.controls
#     q, qdot, ma, mr, mf = states["q"], states["qdot"], states["muscles_ma"], states["muscles_mr"], states["muscles_mf"]
#     tau, muscles = controls["tau"], controls["muscles"]
#
#     # initial and final position
#     np.testing.assert_almost_equal(q[:, 0], np.array((0.07, 1.4)))
#     np.testing.assert_almost_equal(q[:, -1], np.array((1.64470726, 2.25033212)))
#
#     # initial and final velocities
#     np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(qdot[:, -1], np.array((-2.93853331, 3.00564551)))
#
#     # fatigue parameters
#     np.testing.assert_almost_equal(ma[:, 0], np.array((0, 0, 0, 0, 0, 0)))
#     np.testing.assert_almost_equal(
#         ma[:, -1],
#         np.array((0.00739128, 0.00563555, 0.00159309, 0.02418655, 0.02418655, 0.00041913)),
#     )
#     np.testing.assert_almost_equal(mr[:, 0], np.array((1, 1, 1, 1, 1, 1)))
#     np.testing.assert_almost_equal(
#         mr[:, -1], np.array((0.99260018, 0.99281414, 0.99707397, 0.97566527, 0.97566527, 0.99904065))
#     )
#     np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
#     np.testing.assert_almost_equal(
#         mf[:, -1],
#         np.array((8.54868155e-06, 1.55030599e-03, 1.33293886e-03, 1.48176210e-04, 1.48176210e-04, 5.40217808e-04)),
#     )
#
#     # initial and final controls
#     np.testing.assert_almost_equal(tau[:, 0], np.array((0.80920008, 1.66855572)))
#     np.testing.assert_almost_equal(tau[:, -2], np.array((0.81847388, -0.85234628)))
#
#     np.testing.assert_almost_equal(
#         muscles[:, 0],
#         np.array((6.22395441e-08, 4.38966513e-01, 3.80781292e-01, 2.80532298e-07, 2.80532298e-07, 2.26601989e-01)),
#     )
#     np.testing.assert_almost_equal(
#         muscles[:, -2],
#         np.array((8.86069119e-03, 1.17337666e-08, 1.28715148e-08, 2.02340603e-02, 2.02340603e-02, 2.16517945e-08)),
#     )
#
#     # save and load
#     TestUtils.save_and_load(sol, ocp, True)
#
#     # simulate
#     TestUtils.simulate(sol)
#
#
# def test_michaud_fatigable_muscles():
#     from bioptim.examples.fatigue import static_arm_with_fatigue as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#
#     model_path = f"{bioptim_folder}/models/arm26_constant.bioMod"
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=model_path,
#         final_time=0.9,
#         n_shooting=5,
#         fatigue_type="michaud",
#         ode_solver=OdeSolver.COLLOCATION(),
#         torque_level=1,
#     )
#     sol = ocp.solve()
#
#     # Check objective function value
#     f = np.array(sol.cost)
#     np.testing.assert_equal(f.shape, (1, 1))
#     if platform.system() == "Linux":
#         np.testing.assert_almost_equal(f[0, 0], 16.32389073)
#
#     # Check constraints
#     g = np.array(sol.constraints)
#     np.testing.assert_equal(g.shape, (702, 1))
#     np.testing.assert_almost_equal(g, np.zeros((702, 1)))
#
#     # Check some of the results
#     states, controls = sol.states, sol.controls
#     q, qdot, ma, mr, mf = states["q"], states["qdot"], states["muscles_ma"], states["muscles_mr"], states["muscles_mf"]
#     tau, muscles = controls["tau"], controls["muscles"]
#
#     # initial and final position
#     np.testing.assert_almost_equal(q[:, 0], np.array((0.07, 1.4)))
#     np.testing.assert_almost_equal(q[:, -1], np.array((1.64470726, 2.25033212)))
#
#     # initial and final velocities
#     np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(ma[:, 0], np.array((0, 0, 0, 0, 0, 0)))
#     np.testing.assert_almost_equal(mr[:, 0], np.array((1, 1, 1, 1, 1, 1)))
#     np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
#     np.testing.assert_almost_equal(
#         mf[:, -1],
#         np.array((-9.99967420e-09, 5.94635926e-05, 4.24565569e-05, -9.99959286e-09, -9.99952496e-09, -9.82393782e-09)),
#     )
#     if platform.system() == "Linux":
#         np.testing.assert_almost_equal(qdot[:, -1], np.array((-3.89135683, 3.68787547)))
#         np.testing.assert_almost_equal(
#             ma[:, -1], np.array((0.03924825, 0.01089096, 0.00208433, 0.05019895, 0.05019891, 0.00058203))
#         )
#         np.testing.assert_almost_equal(
#             mr[:, -1], np.array((0.96071397, 0.98825271, 0.9973155, 0.94968454, 0.94968458, 0.99917771))
#         )
#         np.testing.assert_almost_equal(tau[:, 0], np.array((0.96697613, 0.76868865)))
#         np.testing.assert_almost_equal(tau[:, -2], np.array((0.59833568, -0.73455239)))
#         np.testing.assert_almost_equal(
#             muscles[:, 0],
#             np.array((1.46440848e-07, 3.21982748e-01, 2.28408896e-01, 3.72307809e-07, 3.72306603e-07, 1.69987370e-01)),
#         )
#         np.testing.assert_almost_equal(
#             muscles[:, -2],
#             np.array((0.04419817, 0.00474247, 0.00090762, 0.04843387, 0.04843384, 0.00025345)),
#         )
#
#     # save and load
#     TestUtils.save_and_load(sol, ocp, True)
#
#     # simulate
#     TestUtils.simulate(sol)
#
#
# def test_effort_fatigable_muscles():
#     from bioptim.examples.fatigue import static_arm_with_fatigue as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#
#     model_path = f"{bioptim_folder}/models/arm26_constant.bioMod"
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=model_path,
#         final_time=0.9,
#         n_shooting=5,
#         fatigue_type="effort",
#         ode_solver=OdeSolver.COLLOCATION(),
#         torque_level=1,
#     )
#     sol = ocp.solve()
#
#     # Check objective function value
#     f = np.array(sol.cost)
#     np.testing.assert_equal(f.shape, (1, 1))
#     np.testing.assert_almost_equal(f[0, 0], 15.670790035133818)
#
#     # Check constraints
#     g = np.array(sol.constraints)
#     np.testing.assert_equal(g.shape, (252, 1))
#     np.testing.assert_almost_equal(g, np.zeros((252, 1)))
#
#     # Check some of the results
#     states, controls = sol.states, sol.controls
#     q, qdot, mf = states["q"], states["qdot"], states["muscles_mf"]
#     tau, muscles = controls["tau"], controls["muscles"]
#
#     # initial and final position
#     np.testing.assert_almost_equal(q[:, 0], np.array((0.07, 1.4)))
#     np.testing.assert_almost_equal(q[:, -1], np.array((1.64470726, 2.25033212)))
#
#     # initial and final velocities
#     np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(qdot[:, -1], np.array((-3.88775177, 3.63334333)))
#
#     # fatigue parameters
#     np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
#     np.testing.assert_almost_equal(
#         mf[:, -1],
#         np.array((0, 6.37439422e-05, 4.49189030e-05, 0, 0, 0)),
#     )
#
#     # initial and final controls
#     np.testing.assert_almost_equal(tau[:, 0], np.array((1.00151692, 0.75680941)))
#     np.testing.assert_almost_equal(tau[:, -2], np.array((0.52586761, -0.65113307)))
#
#     np.testing.assert_almost_equal(
#         muscles[:, 0],
#         np.array((-3.28714697e-09, 3.22448892e-01, 2.29707231e-01, 2.48558443e-08, 2.48558443e-08, 1.68035326e-01)),
#     )
#     np.testing.assert_almost_equal(
#         muscles[:, -2],
#         np.array((3.86483818e-02, 1.10050313e-09, 2.74222702e-09, 4.25097771e-02, 4.25097771e-02, 6.56233597e-09)),
#     )
#
#     # save and load
#     TestUtils.save_and_load(sol, ocp, True)
#
#     # simulate
#     TestUtils.simulate(sol)
#
#
# def test_fatigable_xia_torque_non_split():
#     from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#
#     model_path = f"{bioptim_folder}/models/pendulum.bioMod"
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=model_path,
#         final_time=1,
#         n_shooting=10,
#         fatigue_type="xia",
#         split_controls=False,
#         use_sx=False,
#     )
#     sol = ocp.solve()
#
#     # Check objective function value
#     f = np.array(sol.cost)
#     np.testing.assert_equal(f.shape, (1, 1))
#     if platform.system() == "Linux":
#         np.testing.assert_almost_equal(f[0, 0], 681.4936347682981)
#
#     # Check constraints
#     g = np.array(sol.constraints)
#     np.testing.assert_equal(g.shape, (160, 1))
#     np.testing.assert_almost_equal(g, np.zeros((160, 1)))
#
#     # Check some of the results
#     states, controls = sol.states, sol.controls
#     q, qdot = states["q"], states["qdot"]
#     ma_minus, mr_minus, mf_minus = states["tau_minus_ma"], states["tau_minus_mr"], states["tau_minus_mf"]
#     ma_plus, mr_plus, mf_plus = states["tau_plus_ma"], states["tau_plus_mr"], states["tau_plus_mf"]
#     tau = controls["tau"]
#
#     # initial and final position
#     np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))
#     np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))
#     np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0.0, 0)))
#     np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
#     np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
#     np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
#
#     if platform.system() == "Linux":
#         np.testing.assert_almost_equal(ma_minus[:, -1], np.array((2.05715389e-01, 0)))
#         np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.71681593, 1)))
#         np.testing.assert_almost_equal(mf_minus[:, -1], np.array((7.74686771e-02, 0)))
#         np.testing.assert_almost_equal(ma_plus[:, -1], np.array((4.54576950e-03, 0)))
#         np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.91265673, 1)))
#         np.testing.assert_almost_equal(mf_plus[:, -1], np.array((8.27975034e-02, 0)))
#         np.testing.assert_almost_equal(tau[:, 0], np.array((4.65387493, 0)))
#         np.testing.assert_almost_equal(tau[:, -2], np.array((-21.7531631, 0)))
#
#     # save and load
#     TestUtils.save_and_load(sol, ocp, True)
#
#     # simulate
#     TestUtils.simulate(sol)
#
#
# def test_fatigable_xia_torque_split():
#     from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#
#     model_path = f"{bioptim_folder}/models/pendulum.bioMod"
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=model_path, final_time=1, n_shooting=30, fatigue_type="xia", split_controls=True, use_sx=False
#     )
#     sol = ocp.solve()
#
#     # Check objective function value
#     f = np.array(sol.cost)
#     np.testing.assert_equal(f.shape, (1, 1))
#     np.testing.assert_almost_equal(f[0, 0], 46.97293026598778)
#
#     # Check constraints
#     g = np.array(sol.constraints)
#     np.testing.assert_equal(g.shape, (480, 1))
#     np.testing.assert_almost_equal(g, np.zeros((480, 1)))
#
#     # Check some of the results
#     states, controls = sol.states, sol.controls
#     q, qdot = states["q"], states["qdot"]
#     ma_minus, mr_minus, mf_minus = states["tau_minus_ma"], states["tau_minus_mr"], states["tau_minus_mf"]
#     ma_plus, mr_plus, mf_plus = states["tau_plus_ma"], states["tau_plus_mr"], states["tau_plus_mf"]
#     tau_minus, tau_plus = controls["tau_minus"], controls["tau_plus"]
#
#     # initial and final position
#     np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))
#
#     np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))
#
#     np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0.0, 0)))
#     np.testing.assert_almost_equal(ma_minus[:, -1], np.array((9.74835527e-02, 0)))
#     np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
#     np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.88266826, 1)))
#     np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(mf_minus[:, -1], np.array((1.98481921e-02, 0)))
#     np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(ma_plus[:, -1], np.array((5.69110401e-06, 0)))
#     np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
#     np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.9891588, 1)))
#     np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(mf_plus[:, -1], np.array((1.08355110e-02, 0)))
#
#     np.testing.assert_almost_equal(tau_minus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-10.29111867, 0)))
#     np.testing.assert_almost_equal(tau_plus[:, 0], np.array((7.0546191, 0)))
#     np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))
#
#     # save and load
#     TestUtils.save_and_load(sol, ocp, True)
#
#     # simulate
#     TestUtils.simulate(sol)
#
#
# def test_fatigable_xia_stabilized_torque_split():
#     from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#
#     model_path = f"{bioptim_folder}/models/pendulum.bioMod"
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=model_path,
#         final_time=1,
#         n_shooting=30,
#         fatigue_type="xia_stabilized",
#         split_controls=True,
#         use_sx=False,
#     )
#     sol = ocp.solve()
#
#     # Check objective function value
#     f = np.array(sol.cost)
#     np.testing.assert_equal(f.shape, (1, 1))
#     np.testing.assert_almost_equal(f[0, 0], 46.97293026598767)
#
#     # Check constraints
#     g = np.array(sol.constraints)
#     np.testing.assert_equal(g.shape, (480, 1))
#     np.testing.assert_almost_equal(g, np.zeros((480, 1)))
#
#     # Check some of the results
#     states, controls = sol.states, sol.controls
#     q, qdot = states["q"], states["qdot"]
#     ma_minus, mr_minus, mf_minus = states["tau_minus_ma"], states["tau_minus_mr"], states["tau_minus_mf"]
#     ma_plus, mr_plus, mf_plus = states["tau_plus_ma"], states["tau_plus_mr"], states["tau_plus_mf"]
#     tau_minus, tau_plus = controls["tau_minus"], controls["tau_plus"]
#
#     # initial and final position
#     np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))
#
#     np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))
#
#     np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0.0, 0)))
#     np.testing.assert_almost_equal(ma_minus[:, -1], np.array((9.74835527e-02, 0)))
#     np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
#     np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.88266826, 1)))
#     np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(mf_minus[:, -1], np.array((1.98481921e-02, 0)))
#     np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(ma_plus[:, -1], np.array((5.69110401e-06, 0)))
#     np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
#     np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.9891588, 1)))
#     np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(mf_plus[:, -1], np.array((1.08355110e-02, 0)))
#
#     np.testing.assert_almost_equal(tau_minus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-10.29111867, 0)))
#     np.testing.assert_almost_equal(tau_plus[:, 0], np.array((7.0546191, 0)))
#     np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))
#
#     # save and load
#     TestUtils.save_and_load(sol, ocp, True)
#
#     # simulate
#     TestUtils.simulate(sol)
#
#
# def test_fatigable_michaud_torque_non_split():
#     from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#
#     model_path = f"{bioptim_folder}/models/pendulum.bioMod"
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=model_path,
#         final_time=1,
#         n_shooting=10,
#         fatigue_type="michaud",
#         split_controls=False,
#         use_sx=False,
#     )
#     sol = ocp.solve()
#
#     # Check objective function value
#     f = np.array(sol.cost)
#     np.testing.assert_equal(f.shape, (1, 1))
#     if platform.system() == "Linux":
#         np.testing.assert_almost_equal(f[0, 0], 249.6633124854865)
#
#     # Check constraints
#     g = np.array(sol.constraints)
#     np.testing.assert_equal(g.shape, (200, 1))
#     np.testing.assert_almost_equal(g, np.zeros((200, 1)), decimal=6)
#
#     # Check some of the results
#     states, controls = sol.states, sol.controls
#     q, qdot = states["q"], states["qdot"]
#     ma_minus, mr_minus, mf_minus = states["tau_minus_ma"], states["tau_minus_mr"], states["tau_minus_mf"]
#     ma_plus, mr_plus, mf_plus = states["tau_plus_ma"], states["tau_plus_mr"], states["tau_plus_mf"]
#     tau = controls["tau"]
#
#     # initial and final position
#     np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))
#
#     np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))
#
#     np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
#     np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
#     np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
#
#     if platform.system() == "Linux":
#         np.testing.assert_almost_equal(ma_minus[:, -1], np.array((1.41407692e-01, 0)))
#         np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.85829437, 1)))
#         np.testing.assert_almost_equal(mf_minus[:, -1], np.array((0, 0)))
#         np.testing.assert_almost_equal(ma_plus[:, -1], np.array((1.39510468e-03, 0)))
#         np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.9982828, 1)))
#         np.testing.assert_almost_equal(mf_plus[:, -1], np.array((1.76513566e-05, 0)))
#         np.testing.assert_almost_equal(tau[:, 0], np.array((6.24822558, 0)))
#         np.testing.assert_almost_equal(tau[:, -2], np.array((-14.19965472, 0)))
#
#     # save and load
#     TestUtils.save_and_load(sol, ocp, True)
#
#     # simulate
#     TestUtils.simulate(sol, decimal_value=5)
#
#
# def test_fatigable_michaud_torque_split():
#     from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#
#     model_path = f"{bioptim_folder}/models/pendulum.bioMod"
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=model_path,
#         final_time=1,
#         n_shooting=10,
#         fatigue_type="michaud",
#         split_controls=True,
#         use_sx=False,
#     )
#     sol = ocp.solve()
#
#     # Check objective function value
#     f = np.array(sol.cost)
#     np.testing.assert_equal(f.shape, (1, 1))
#     np.testing.assert_almost_equal(f[0, 0], 66.4869989782804)
#
#     # Check constraints
#     g = np.array(sol.constraints)
#     np.testing.assert_equal(g.shape, (200, 1))
#     np.testing.assert_almost_equal(g, np.zeros((200, 1)))
#
#     # Check some of the results
#     states, controls = sol.states, sol.controls
#     q, qdot = states["q"], states["qdot"]
#     ma_minus, mr_minus, mf_minus = states["tau_minus_ma"], states["tau_minus_mr"], states["tau_minus_mf"]
#     ma_plus, mr_plus, mf_plus = states["tau_plus_ma"], states["tau_plus_mr"], states["tau_plus_mf"]
#     tau_minus, tau_plus = controls["tau_minus"], controls["tau_plus"]
#
#     # initial and final position
#     np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))
#
#     np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))
#
#     np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(ma_minus[:, -1], np.array((1.14840287e-01, 0)))
#     np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
#     np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.88501154, 1)))
#     np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(mf_minus[:, -1], np.array((0, 0)))
#     np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(ma_plus[:, -1], np.array((6.06085673e-04, 0)))
#     np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
#     np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.99924023, 1)))
#     np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(mf_plus[:, -1], np.array((0, 0)))
#
#     np.testing.assert_almost_equal(tau_minus[:, 0], np.array((-2.39672721e-07, 0)))
#     np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-11.53208375, 0)))
#     if platform.system() == "Linux":
#         np.testing.assert_almost_equal(tau_plus[:, 0], np.array((5.03417919, 0)))
#     np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))
#
#     # save and load
#     TestUtils.save_and_load(sol, ocp, True)
#
#     # simulate
#     TestUtils.simulate(sol, decimal_value=6)
#
#
# def test_fatigable_effort_torque_non_split():
#     from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#
#     model_path = f"{bioptim_folder}/models/pendulum.bioMod"
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=model_path,
#         final_time=1,
#         n_shooting=10,
#         fatigue_type="effort",
#         split_controls=False,
#         use_sx=False,
#     )
#     sol = ocp.solve()
#
#     # Check objective function value
#     f = np.array(sol.cost)
#     np.testing.assert_equal(f.shape, (1, 1))
#     if platform.system() == "Linux":
#         np.testing.assert_almost_equal(f[0, 0], 758.5267850888707)
#
#     # Check constraints
#     g = np.array(sol.constraints)
#     np.testing.assert_equal(g.shape, (80, 1))
#     np.testing.assert_almost_equal(g, np.zeros((80, 1)))
#
#     # Check some of the results
#     states, controls = sol.states, sol.controls
#     q, qdot = states["q"], states["qdot"]
#     mf_minus, mf_plus = states["tau_minus_mf"], states["tau_plus_mf"]
#     tau = controls["tau"]
#
#     # initial and final position
#     np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))
#
#     np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))
#
#     np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
#     np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
#
#     if platform.system() == "Linux":
#         np.testing.assert_almost_equal(mf_minus[:, -1], np.array((9.83471568e-05, 1.99600599e-06)))
#         np.testing.assert_almost_equal(mf_plus[:, -1], np.array((9.03716040e-05, 0)))
#         np.testing.assert_almost_equal(tau[:, 0], np.array((4.97692313, 0)))
#         np.testing.assert_almost_equal(tau[:, -2], np.array((-22.22043242, 0)))
#
#     # save and load
#     TestUtils.save_and_load(sol, ocp, True)
#
#     # simulate
#     TestUtils.simulate(sol)
#
#
# def test_fatigable_effort_torque_split():
#     from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#
#     model_path = f"{bioptim_folder}/models/pendulum.bioMod"
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=model_path,
#         final_time=1,
#         n_shooting=10,
#         fatigue_type="effort",
#         split_controls=True,
#         use_sx=False,
#     )
#     sol = ocp.solve()
#
#     # Check objective function value
#     if platform.system() == "Linux":
#         f = np.array(sol.cost)
#         np.testing.assert_equal(f.shape, (1, 1))
#         np.testing.assert_almost_equal(f[0, 0], 124.09811263203727)
#
#         # Check constraints
#         g = np.array(sol.constraints)
#         np.testing.assert_equal(g.shape, (80, 1))
#         np.testing.assert_almost_equal(g, np.zeros((80, 1)))
#
#         # Check some of the results
#         states, controls = sol.states, sol.controls
#         q, qdot = states["q"], states["qdot"]
#         mf_minus, mf_plus = states["tau_minus_mf"], states["tau_plus_mf"]
#         tau_minus, tau_plus = controls["tau_minus"], controls["tau_plus"]
#
#         # initial and final position
#         np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
#         np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))
#
#         np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
#         np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))
#
#         np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
#         np.testing.assert_almost_equal(mf_minus[:, -1], np.array((4.51209384e-05, 1.99600599e-06)))
#         np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
#         np.testing.assert_almost_equal(mf_plus[:, -1], np.array((4.31950457e-05, 0)))
#
#         np.testing.assert_almost_equal(tau_minus[:, 0], np.array((-8.39444342e-08, 0)))
#         np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-12.03087219, 0)))
#         np.testing.assert_almost_equal(tau_plus[:, 0], np.array((5.85068579, 0)))
#         np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))
#
#     # save and load
#     TestUtils.save_and_load(sol, ocp, True)
#
#     # simulate
#     TestUtils.simulate(sol)

import numpy as np
from bioptim import OdeSolver

from .utils import TestUtils


def test_xia_fatigable_muscles():
    from bioptim.examples.fatigue import static_arm_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/arm26_constant.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=0.9,
        n_shooting=5,
        fatigue_type="xia",
        ode_solver=OdeSolver.COLLOCATION(),
        torque_level=1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 19.770521758810368)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (552, 1))
    np.testing.assert_almost_equal(g, np.zeros((552, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot, ma, mr, mf = states["q"], states["qdot"], states["muscles_ma"], states["muscles_mr"], states["muscles_mf"]
    tau, muscles = controls["tau"], controls["muscles"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.07, 1.4)))
    np.testing.assert_almost_equal(q[:, -1], np.array((1.64470726, 2.25033212)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-2.93853331, 3.00564551)))

    # fatigue parameters
    np.testing.assert_almost_equal(ma[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        ma[:, -1], np.array((0.00739128, 0.00563555, 0.00159309, 0.02418655, 0.02418655, 0.00041913))
    )
    np.testing.assert_almost_equal(mr[:, 0], np.array((1, 1, 1, 1, 1, 1)))
    np.testing.assert_almost_equal(
        mr[:, -1], np.array((0.99260018, 0.99281414, 0.99707397, 0.97566527, 0.97566527, 0.99904065))
    )
    np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        mf[:, -1],
        np.array((8.54868154e-06, 1.55030599e-03, 1.33293886e-03, 1.48176210e-04, 1.48176210e-04, 5.40217808e-04)),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.80920008, 1.66855572)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.81847388, -0.85234628)))

    np.testing.assert_almost_equal(
        muscles[:, 0],
        np.array((6.22395441e-08, 4.38966513e-01, 3.80781292e-01, 2.80532297e-07, 2.80532297e-07, 2.26601989e-01)),
    )
    np.testing.assert_almost_equal(
        muscles[:, -2],
        np.array((8.86069119e-03, 1.17337666e-08, 1.28715148e-08, 2.02340603e-02, 2.02340603e-02, 2.16517945e-088)),
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_xia_stabilized_fatigable_muscles():
    from bioptim.examples.fatigue import static_arm_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/arm26_constant.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=0.9,
        n_shooting=5,
        fatigue_type="xia_stabilized",
        ode_solver=OdeSolver.COLLOCATION(),
        torque_level=1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 19.770521758810393)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (552, 1))
    np.testing.assert_almost_equal(g, np.zeros((552, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot, ma, mr, mf = states["q"], states["qdot"], states["muscles_ma"], states["muscles_mr"], states["muscles_mf"]
    tau, muscles = controls["tau"], controls["muscles"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.07, 1.4)))
    np.testing.assert_almost_equal(q[:, -1], np.array((1.64470726, 2.25033212)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-2.93853331, 3.00564551)))

    # fatigue parameters
    np.testing.assert_almost_equal(ma[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        ma[:, -1],
        np.array((0.00739128, 0.00563555, 0.00159309, 0.02418655, 0.02418655, 0.00041913)),
    )
    np.testing.assert_almost_equal(mr[:, 0], np.array((1, 1, 1, 1, 1, 1)))
    np.testing.assert_almost_equal(
        mr[:, -1], np.array((0.99260018, 0.99281414, 0.99707397, 0.97566527, 0.97566527, 0.99904065))
    )
    np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        mf[:, -1],
        np.array((8.54868155e-06, 1.55030599e-03, 1.33293886e-03, 1.48176210e-04, 1.48176210e-04, 5.40217808e-04)),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.80920008, 1.66855572)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.81847388, -0.85234628)))

    np.testing.assert_almost_equal(
        muscles[:, 0],
        np.array((6.22395441e-08, 4.38966513e-01, 3.80781292e-01, 2.80532298e-07, 2.80532298e-07, 2.26601989e-01)),
    )
    np.testing.assert_almost_equal(
        muscles[:, -2],
        np.array((8.86069119e-03, 1.17337666e-08, 1.28715148e-08, 2.02340603e-02, 2.02340603e-02, 2.16517945e-08)),
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_michaud_fatigable_muscles():
    from bioptim.examples.fatigue import static_arm_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/arm26_constant.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=0.9,
        n_shooting=5,
        fatigue_type="michaud",
        ode_solver=OdeSolver.COLLOCATION(),
        torque_level=1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if platform.system() == "Linux":
        np.testing.assert_almost_equal(f[0, 0], 16.32389073)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (702, 1))
    np.testing.assert_almost_equal(g, np.zeros((702, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot, ma, mr, mf = states["q"], states["qdot"], states["muscles_ma"], states["muscles_mr"], states["muscles_mf"]
    tau, muscles = controls["tau"], controls["muscles"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.07, 1.4)))
    np.testing.assert_almost_equal(q[:, -1], np.array((1.64470726, 2.25033212)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(mr[:, 0], np.array((1, 1, 1, 1, 1, 1)))
    np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        mf[:, -1],
        np.array((-9.99967420e-09, 5.94635926e-05, 4.24565569e-05, -9.99959286e-09, -9.99952496e-09, -9.82393782e-09)),
    )
    if platform.system() == "Linux":
        np.testing.assert_almost_equal(qdot[:, -1], np.array((-3.89135683, 3.68787547)))
        np.testing.assert_almost_equal(
            ma[:, -1], np.array((0.03924825, 0.01089096, 0.00208433, 0.05019895, 0.05019891, 0.00058203))
        )
        np.testing.assert_almost_equal(
            mr[:, -1], np.array((0.96071397, 0.98825271, 0.9973155, 0.94968454, 0.94968458, 0.99917771))
        )
        np.testing.assert_almost_equal(tau[:, 0], np.array((0.96697613, 0.76868865)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((0.59833568, -0.73455239)))
        np.testing.assert_almost_equal(
            muscles[:, 0],
            np.array((1.46440848e-07, 3.21982748e-01, 2.28408896e-01, 3.72307809e-07, 3.72306603e-07, 1.69987370e-01)),
        )
        np.testing.assert_almost_equal(
            muscles[:, -2],
            np.array((0.04419817, 0.00474247, 0.00090762, 0.04843387, 0.04843384, 0.00025345)),
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_effort_fatigable_muscles():
    from bioptim.examples.fatigue import static_arm_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/arm26_constant.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=0.9,
        n_shooting=5,
        fatigue_type="effort",
        ode_solver=OdeSolver.COLLOCATION(),
        torque_level=1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 15.670790035133818)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (252, 1))
    np.testing.assert_almost_equal(g, np.zeros((252, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot, mf = states["q"], states["qdot"], states["muscles_mf"]
    tau, muscles = controls["tau"], controls["muscles"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.07, 1.4)))
    np.testing.assert_almost_equal(q[:, -1], np.array((1.64470726, 2.25033212)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-3.88775177, 3.63334333)))

    # fatigue parameters
    np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        mf[:, -1],
        np.array((0, 6.37439422e-05, 4.49189030e-05, 0, 0, 0)),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.00151692, 0.75680941)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.52586761, -0.65113307)))

    np.testing.assert_almost_equal(
        muscles[:, 0],
        np.array((-3.28714697e-09, 3.22448892e-01, 2.29707231e-01, 2.48558443e-08, 2.48558443e-08, 1.68035326e-01)),
    )
    np.testing.assert_almost_equal(
        muscles[:, -2],
        np.array((3.86483818e-02, 1.10050313e-09, 2.74222702e-09, 4.25097771e-02, 4.25097771e-02, 6.56233597e-09)),
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_fatigable_xia_torque_non_split():
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    # it doesn't pass on macos
    if platform.system() == "Darwin":
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="xia",
        split_controls=False,
        use_sx=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if platform.system() == "Linux":
        np.testing.assert_almost_equal(f[0, 0], 681.4936347682981)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (160, 1))
    np.testing.assert_almost_equal(g, np.zeros((160, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot = states["q"], states["qdot"]
    ma_minus, mr_minus, mf_minus = states["tau_minus_ma"], states["tau_minus_mr"], states["tau_minus_mf"]
    ma_plus, mr_plus, mf_plus = states["tau_plus_ma"], states["tau_plus_mr"], states["tau_plus_mf"]
    tau = controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0.0, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))

    if platform.system() == "Linux":
        np.testing.assert_almost_equal(ma_minus[:, -1], np.array((2.05715389e-01, 0)))
        np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.71681593, 1)))
        np.testing.assert_almost_equal(mf_minus[:, -1], np.array((7.74686771e-02, 0)))
        np.testing.assert_almost_equal(ma_plus[:, -1], np.array((4.54576950e-03, 0)))
        np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.91265673, 1)))
        np.testing.assert_almost_equal(mf_plus[:, -1], np.array((8.27975034e-02, 0)))
        np.testing.assert_almost_equal(tau[:, 0], np.array((4.65387493, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-21.7531631, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_fatigable_xia_torque_split():
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path, final_time=1, n_shooting=30, fatigue_type="xia", split_controls=True, use_sx=False
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 46.97293026598778)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (480, 1))
    np.testing.assert_almost_equal(g, np.zeros((480, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot = states["q"], states["qdot"]
    ma_minus, mr_minus, mf_minus = states["tau_minus_ma"], states["tau_minus_mr"], states["tau_minus_mf"]
    ma_plus, mr_plus, mf_plus = states["tau_plus_ma"], states["tau_plus_mr"], states["tau_plus_mf"]
    tau_minus, tau_plus = controls["tau_minus"], controls["tau_plus"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0.0, 0)))
    np.testing.assert_almost_equal(ma_minus[:, -1], np.array((9.74835527e-02, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.88266826, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_minus[:, -1], np.array((1.98481921e-02, 0)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_plus[:, -1], np.array((5.69110401e-06, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.9891588, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_plus[:, -1], np.array((1.08355110e-02, 0)))

    np.testing.assert_almost_equal(tau_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-10.29111867, 0)))
    np.testing.assert_almost_equal(tau_plus[:, 0], np.array((7.0546191, 0)))
    np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_fatigable_xia_stabilized_torque_split():
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=30,
        fatigue_type="xia_stabilized",
        split_controls=True,
        use_sx=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 46.97293026598767)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (480, 1))
    np.testing.assert_almost_equal(g, np.zeros((480, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot = states["q"], states["qdot"]
    ma_minus, mr_minus, mf_minus = states["tau_minus_ma"], states["tau_minus_mr"], states["tau_minus_mf"]
    ma_plus, mr_plus, mf_plus = states["tau_plus_ma"], states["tau_plus_mr"], states["tau_plus_mf"]
    tau_minus, tau_plus = controls["tau_minus"], controls["tau_plus"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0.0, 0)))
    np.testing.assert_almost_equal(ma_minus[:, -1], np.array((9.74835527e-02, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.88266826, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_minus[:, -1], np.array((1.98481921e-02, 0)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_plus[:, -1], np.array((5.69110401e-06, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.9891588, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_plus[:, -1], np.array((1.08355110e-02, 0)))

    np.testing.assert_almost_equal(tau_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-10.29111867, 0)))
    np.testing.assert_almost_equal(tau_plus[:, 0], np.array((7.0546191, 0)))
    np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_fatigable_michaud_torque_non_split():
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="michaud",
        split_controls=False,
        use_sx=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if platform.system() == "Linux":
        np.testing.assert_almost_equal(f[0, 0], 249.6633124854865)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (200, 1))
    np.testing.assert_almost_equal(g, np.zeros((200, 1)), decimal=6)

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot = states["q"], states["qdot"]
    ma_minus, mr_minus, mf_minus = states["tau_minus_ma"], states["tau_minus_mr"], states["tau_minus_mf"]
    ma_plus, mr_plus, mf_plus = states["tau_plus_ma"], states["tau_plus_mr"], states["tau_plus_mf"]
    tau = controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))

    if platform.system() == "Linux":
        np.testing.assert_almost_equal(ma_minus[:, -1], np.array((1.41407692e-01, 0)))
        np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.85829437, 1)))
        np.testing.assert_almost_equal(mf_minus[:, -1], np.array((0, 0)))
        np.testing.assert_almost_equal(ma_plus[:, -1], np.array((1.39510468e-03, 0)))
        np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.9982828, 1)))
        np.testing.assert_almost_equal(mf_plus[:, -1], np.array((1.76513566e-05, 0)))
        np.testing.assert_almost_equal(tau[:, 0], np.array((6.24822558, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-14.19965472, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol, decimal_value=5)


def test_fatigable_michaud_torque_split():
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="michaud",
        split_controls=True,
        use_sx=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 66.4869989782804)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (200, 1))
    np.testing.assert_almost_equal(g, np.zeros((200, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot = states["q"], states["qdot"]
    ma_minus, mr_minus, mf_minus = states["tau_minus_ma"], states["tau_minus_mr"], states["tau_minus_mf"]
    ma_plus, mr_plus, mf_plus = states["tau_plus_ma"], states["tau_plus_mr"], states["tau_plus_mf"]
    tau_minus, tau_plus = controls["tau_minus"], controls["tau_plus"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_minus[:, -1], np.array((1.14840287e-01, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.88501154, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_minus[:, -1], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_plus[:, -1], np.array((6.06085673e-04, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.99924023, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_plus[:, -1], np.array((0, 0)))

    np.testing.assert_almost_equal(tau_minus[:, 0], np.array((-2.39672721e-07, 0)))
    np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-11.53208375, 0)))
    if platform.system() == "Linux":
        np.testing.assert_almost_equal(tau_plus[:, 0], np.array((5.03417919, 0)))
    np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol, decimal_value=6)


def test_fatigable_effort_torque_non_split():
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="effort",
        split_controls=False,
        use_sx=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if platform.system() == "Linux":
        np.testing.assert_almost_equal(f[0, 0], 758.5267850888707)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (80, 1))
    np.testing.assert_almost_equal(g, np.zeros((80, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot = states["q"], states["qdot"]
    mf_minus, mf_plus = states["tau_minus_mf"], states["tau_plus_mf"]
    tau = controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))

    if platform.system() == "Linux":
        np.testing.assert_almost_equal(mf_minus[:, -1], np.array((9.83471568e-05, 1.99600599e-06)))
        np.testing.assert_almost_equal(mf_plus[:, -1], np.array((9.03716040e-05, 0)))
        np.testing.assert_almost_equal(tau[:, 0], np.array((4.97692313, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-22.22043242, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_fatigable_effort_torque_split():
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="effort",
        split_controls=True,
        use_sx=False,
    )
    sol = ocp.solve()

    # Check objective function value
    if platform.system() == "Linux":
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 124.09811263203727)

        # Check constraints
        g = np.array(sol.constraints)
        np.testing.assert_equal(g.shape, (80, 1))
        np.testing.assert_almost_equal(g, np.zeros((80, 1)))

        # Check some of the results
        states, controls = sol.states, sol.controls
        q, qdot = states["q"], states["qdot"]
        mf_minus, mf_plus = states["tau_minus_mf"], states["tau_plus_mf"]
        tau_minus, tau_plus = controls["tau_minus"], controls["tau_plus"]

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

        np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(mf_minus[:, -1], np.array((4.51209384e-05, 1.99600599e-06)))
        np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(mf_plus[:, -1], np.array((4.31950457e-05, 0)))

        np.testing.assert_almost_equal(tau_minus[:, 0], np.array((-8.39444342e-08, 0)))
        np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-12.03087219, 0)))
        np.testing.assert_almost_equal(tau_plus[:, 0], np.array((5.85068579, 0)))
        np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)
