"""
Test for file IO
"""
import os
import pytest

import numpy as np
import biorbd_casadi as biorbd
from bioptim import OdeSolver

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
    n_shooting = 5
    use_residual_torque = True

    # Generate random data to fit
    np.random.seed(42)
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
        np.testing.assert_equal(g.shape, (20 * 5, 1))
        np.testing.assert_almost_equal(g, np.zeros((20 * 5, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (20, 1))
        np.testing.assert_almost_equal(g, np.zeros((20, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau, mus = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.controls["muscles"]

    if ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 1.6168687925491806e-08)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-9.27332670e-06, -1.92130847e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.13095414, -0.52519196]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0003876, 0.00011132]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-3.48898607, 5.69081695]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([9.20045224e-07, -1.43441728e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([1.02698339e-06, -1.45090939e-06]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37453715, 0.95067227, 0.73198611, 0.5986687, 0.15604161, 0.15601083])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.45602115, 0.78518567, 0.19968753, 0.51422336, 0.59240291, 0.04651572])
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 1.6168383151995866e-08)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-9.26023358e-06, -1.95272568e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.13095416, -0.52519198]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00038863, 0.00010772]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-3.4889827, 5.69080853]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([9.11529554e-07, -1.41077143e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([1.03121849e-06, -1.46222274e-06]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37453703, 0.9506723, 0.73198617, 0.5986685, 0.15604142, 0.15601076])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.45602088, 0.78518577, 0.19968758, 0.51422328, 0.59240284, 0.04651587])
        )

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 3.624795808383824e-08)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-9.10806301e-06, -1.53310138e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.13095467, -0.52519196]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00036361, 0.00013356]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-3.4890094, 5.69101925]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([1.04127177e-06, -1.80879296e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([9.55112770e-07, -1.31205473e-06]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37453927, 0.95067167, 0.73198512, 0.59867185, 0.1560445, 0.15601199])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.45602514, 0.78518424, 0.19968701, 0.51422439, 0.59240386, 0.04651392])
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
    n_shooting = 5
    use_residual_torque = False

    # Generate random data to fit
    np.random.seed(42)
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
    np.testing.assert_almost_equal(f[0, 0], 3.634248634056222e-08)

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (20 * 5, 1))
        np.testing.assert_almost_equal(g, np.zeros((20 * 5, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (20, 1))
        np.testing.assert_almost_equal(g, np.zeros((20, 1)), decimal=6)

    # Check some of the results
    q, qdot, mus = sol.states["q"], sol.states["qdot"], sol.controls["muscles"]

    if ode_solver == OdeSolver.IRK:
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-9.27680641e-06, -1.96170122e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.13095414, -0.52519198]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00038904, 0.00010936]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-3.48898818, 5.69081901]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37453701, 0.95067231, 0.73198617, 0.5986685, 0.15604143, 0.15601076])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.45602116, 0.78518566, 0.19968753, 0.51422337, 0.59240292, 0.0465157])
        )
    elif ode_solver == OdeSolver.COLLOCATION:
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-9.27680641e-06, -1.96170122e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.13095414, -0.52519198]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00039019, 0.00010557]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-3.48898481, 5.69081085]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37453699, 0.95067231, 0.73198618, 0.59866847, 0.15604139, 0.15601075])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.4560209, 0.78518576, 0.19968757, 0.51422329, 0.59240285, 0.04651584])
        )

    elif ode_solver == OdeSolver.RK4:
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-9.11977097e-06, -1.55461431e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.13095464, -0.52519197]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00036513, 0.0001314]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-3.48901148, 5.69102134]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37453916, 0.9506717, 0.73198517, 0.59867169, 0.15604435, 0.15601192])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.45602515, 0.78518424, 0.199687, 0.51422441, 0.59240387, 0.0465139])
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
    final_time = 0.1
    n_shooting = 5

    # Generate random data to fit
    np.random.seed(42)
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
    np.random.seed(42)
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
