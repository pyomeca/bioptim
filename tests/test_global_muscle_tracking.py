"""
Test for file IO
"""
import os
import pytest

import numpy as np
import biorbd_casadi as biorbd
from bioptim import OdeSolver, Solver

from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_muscle_activations_and_states_tracking(ode_solver):
    # Load muscle_activations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_activations_tracker as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    model_path = bioptim_folder + "/models/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.1
    n_shooting = 10
    use_residual_torque = True

    # Generate random data to fit
    np.random.seed(10)
    t, markers_ref, x_ref, muscle_activations_ref = ocp_module.generate_data(
        biorbd_model, final_time, n_shooting, use_residual_torque=use_residual_torque
    )

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = ocp_module.prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_activations_ref,
        x_ref[: biorbd_model.nbQ(), :],
        use_residual_torque=use_residual_torque,
        kin_data_to_track="q",
        ode_solver=ode_solver(),
    )
    solver = Solver.IPOPT()
    # solver.set_maximum_iterations(10)
    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (40 * 5, 1))
        np.testing.assert_almost_equal(g, np.zeros((40 * 5, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (40, 1))
        np.testing.assert_almost_equal(g, np.zeros((40, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau, mus = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.controls["muscles"]

    if ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 2.847493171046114)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.03035811, -0.09255799]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.15947451, -0.48817443]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.42720259, -0.26423124]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([3.39388909, -11.56665709]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-0.01317384,  0.05689357]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([0.1671349,  -0.46822556]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([4.30591315e-01, 6.34995244e-02, 7.38966935e-01, 5.93401437e-01,
 4.31160671e-01, 1.52105655e-04])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([4.11690249e-07, 9.99997822e-01, 9.99995159e-01, 3.03197474e-05,
 5.72835945e-01, 9.99998918e-01])
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 2.8474936844260927)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.03035818, -0.09255773]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.15947462, -0.48817444]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.42700495, -0.2640704]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([3.3938847,  -11.56665503]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-0.01321899,  0.0570118]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([0.16713482, -0.46822536]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.43037341, 0.0635237,  0.7391545,  0.59306444, 0.43101405, 0.00078317])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([4.11717030e-07, 9.99997819e-01, 9.99995141e-01, 3.29313263e-05, 5.72835982e-01, 9.99998917e-01])
        )

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 2.847198941431199)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.03036634, -0.09244015]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.15947144, -0.48817532]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.43166183, -0.25927519]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([3.39397709, -11.56665469]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-0.01360646,  0.05850246]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([0.16713538, -0.46822598]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([4.56339650e-01, 6.44720155e-02, 7.41920893e-01, 5.88857159e-01,
 4.29182691e-01, 1.15932905e-04])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([4.11689891e-07, 9.99997822e-01, 9.99995160e-01, 3.03190037e-05,
 5.72834539e-01, 9.99998918e-01])
        )

    else:
        raise ValueError("Test not implemented")

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=5)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_muscle_activation_no_residual_torque_and_markers_tracking(ode_solver):
    # Load muscle_activations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_activations_tracker as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    model_path = bioptim_folder + "/models/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.1
    n_shooting = 10
    use_residual_torque = False

    # Generate random data to fit
    np.random.seed(10)
    t, markers_ref, x_ref, muscle_activations_ref = ocp_module.generate_data(
        biorbd_model, final_time, n_shooting, use_residual_torque=use_residual_torque
    )

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = ocp_module.prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_activations_ref,
        x_ref[: biorbd_model.nbQ(), :],
        use_residual_torque=use_residual_torque,
        kin_data_to_track="q",
        ode_solver=ode_solver(),
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (40 * 5, 1))
        np.testing.assert_almost_equal(g, np.zeros((40 * 5, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (40, 1))
        np.testing.assert_almost_equal(g, np.zeros((40, 1)), decimal=6)

    # Check some of the results
    q, qdot, mus = sol.states["q"], sol.states["qdot"], sol.controls["muscles"]
    print("f", f[0,0],"\n",
         "q", q[:, 0], q[:, -1],"\n",
          "qdot", qdot[:, 0], qdot[:, -1],"\n"
          "mus", mus[:, 0], mus[:, -2],"\n")

    if ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 2.850233712238023)
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.03036667, -0.09258569]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.15917065, -0.48743124]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.42178169, -0.2601654]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([3.31588211, -11.35093943]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([4.24834961e-01, 6.31395546e-02, 7.43974088e-01, 5.84356397e-01,
 4.27223492e-01, 2.15052183e-04])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([4.10095531e-07, 9.99997830e-01, 9.99995167e-01, 3.20988918e-05,
 5.71432010e-01, 9.99998923e-01])
        )
    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 2.8502337134173876)
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.03036667, -0.09258569]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.15917065, -0.48743124]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.42178129, -0.26016508]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([3.31588202, -11.35093912]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([4.24834527e-01, 6.31395690e-02, 7.43974465e-01, 5.84355722e-01,
 4.27223199e-01, 2.16258140e-04])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([4.10066202e-07, 9.99997831e-01, 9.99995167e-01, 3.20899584e-05,
 5.71432064e-01, 9.99998923e-01])
        )

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 2.8499418137945383)
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.03037517, -0.09246922]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.15916755, -0.48743222]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.42619834, -0.25504332]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([3.31597172, -11.35093665]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([4.50470744e-01, 6.41232299e-02, 7.47093851e-01, 5.79530720e-01,
 4.25122774e-01, 1.55360547e-04])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([4.10096770e-07, 9.99997830e-01, 9.99995167e-01, 3.20980119e-05,
 5.71430484e-01, 9.99998923e-01])
        )
    else:
        raise ValueError("Test not ready")

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=6)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_muscle_excitation_with_torque_and_markers_tracking(ode_solver):
    # Load muscle_excitations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_excitations_tracker as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    model_path = bioptim_folder + "/models/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.2
    n_shooting = 20

    # Generate random data to fit
    np.random.seed(10)
    t, markers_ref, x_ref, muscle_excitations_ref = ocp_module.generate_data(biorbd_model, final_time, n_shooting)

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = ocp_module.prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: biorbd_model.nbQ(), :].T,
        use_residual_torque=True,
        kin_data_to_track="markers",
        ode_solver=ode_solver(),
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (50 * 5, 1))
        np.testing.assert_almost_equal(g, np.zeros((50 * 5, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (50, 1))
        np.testing.assert_almost_equal(g, np.zeros((50, 1)), decimal=6)

    # Check some of the results
    q, qdot, mus_states, tau, mus_controls = (
        sol.states["q"],
        sol.states["qdot"],
        sol.states["muscles"],
        sol.controls["tau"],
        sol.controls["muscles"],
    )

    if ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 4.008048594676488e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.0034705, 0.01684031]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.17038944, -0.76762623]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.03547075, -3.92740332]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.92301008, 1.78684668]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.5126714, 0.69948007, 0.4039512, 0.48028695, 0.537475, 0.3144467])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([5.40602693e-05, -3.43792358e-04]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-2.45516586e-05, -4.60440912e-05]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37747088, 0.95057098, 0.73173072, 0.60097506, 0.15936616, 0.15866445])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.45605498, 0.78523283, 0.19974774, 0.51404221, 0.59223931, 0.04661602])
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 4.0081629979408034e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00346988, 0.0168371]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.17038802, -0.76762245]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.0353528, -3.92691666]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.92303481, 1.78693516]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51266712, 0.69947546, 0.40395133, 0.48028832, 0.53747684, 0.31444732])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([5.40731145e-05, -3.43788410e-04]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-2.45153875e-05, -4.61335015e-05]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37748314, 0.95045862, 0.73171724, 0.600967, 0.15944509, 0.1586948])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.45605398, 0.78522824, 0.19974877, 0.51404386, 0.59223952, 0.04661861])
        )

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 3.9794313261046896e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00345137, 0.01676553]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.17039042, -0.76764697]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.03373366, -3.92126655]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.92313029, 1.78640863]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51268549, 0.69947117, 0.40394632, 0.48027953, 0.53746661, 0.31446889])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([5.37097242e-05, -3.42250644e-04]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-2.45733125e-05, -4.55068941e-05]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.3774798, 0.95056816, 0.73173097, 0.60096548, 0.15935417, 0.15854798])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.4560568, 0.78523111, 0.19974709, 0.51404432, 0.59224108, 0.04661467])
        )

    else:
        raise ValueError("Test not ready")

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_muscle_excitation_no_residual_torque_and_markers_tracking(ode_solver):
    # Load muscle_excitations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_excitations_tracker as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    model_path = bioptim_folder + "/models/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.1
    n_shooting = 5

    # Generate random data to fit
    np.random.seed(10)
    t, markers_ref, x_ref, muscle_excitations_ref = ocp_module.generate_data(biorbd_model, final_time, n_shooting)

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = ocp_module.prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: biorbd_model.nbQ(), :].T,
        use_residual_torque=False,
        kin_data_to_track="markers",
        ode_solver=ode_solver(),
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (50 * 5, 1))
        np.testing.assert_almost_equal(g, np.zeros((50 * 5, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (50, 1))
        np.testing.assert_almost_equal(g, np.zeros((50, 1)), decimal=6)

    # Check some of the results
    q, qdot, mus_states, mus_controls = (
        sol.states["q"],
        sol.states["qdot"],
        sol.states["muscles"],
        sol.controls["muscles"],
    )

    if ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 4.0100479928729196e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.0034751, 0.01685431]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.17038791, -0.76762034]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.03586838, -3.92859333]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.92300111, 1.78692532]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51266794, 0.69948076, 0.40395187, 0.48028633, 0.53747424, 0.31444829])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37746726, 0.9505724, 0.73173087, 0.60097671, 0.15936876, 0.15866727])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.45605437, 0.78523337, 0.1997479, 0.51404166, 0.59223881, 0.04661636])
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 4.0101623987442914e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00347448, 0.0168511]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.17038649, -0.76761656]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.03575054, -3.92810697]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.92302588, 1.78701392]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51266365, 0.69947614, 0.403952, 0.4802877, 0.53747608, 0.31444892])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37747952, 0.95046004, 0.73171739, 0.60096865, 0.15944769, 0.15869762])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.45605336, 0.78522878, 0.19974893, 0.51404331, 0.59223902, 0.04661895])
        )

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 3.981409409342779e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00345594, 0.01677944]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.17038891, -0.76764111]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.03412935, -3.92245047]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.92312113, 1.78648617]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51268205, 0.69947185, 0.40394699, 0.48027891, 0.53746585, 0.31447048])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37747623, 0.95056957, 0.73173112, 0.60096712, 0.15935676, 0.15855066])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.45605619, 0.78523164, 0.19974724, 0.51404377, 0.59224058, 0.04661501])
        )

    else:
        raise ValueError("Test not implemented")

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)
