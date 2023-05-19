"""
Test for file IO
"""
import os
import platform
import pytest
import re

import numpy as np
from bioptim import (
    BiorbdModel,
    ConstraintList,
    ConstraintFcn,
    DynamicsList,
    DynamicsFcn,
    InitialGuessList,
    BoundsList,
    Node,
    ObjectiveList,
    ObjectiveFcn,
    OptimalControlProgram,
    OdeSolver,
)

from .utils import TestUtils


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_pendulum_min_time_mayer(ode_solver, assume_phase_dynamics):
    # Load pendulum_min_time_Mayer
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Mayer as ocp_module

    # For reducing time assume_phase_dynamics=False is skipped for redundant tests
    if not assume_phase_dynamics and ode_solver == OdeSolver.COLLOCATION:
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    if ode_solver == OdeSolver.IRK:
        ft = 2
        ns = 35
    elif ode_solver == OdeSolver.COLLOCATION:
        ft = 2
        ns = 10
    elif ode_solver == OdeSolver.RK4:
        ft = 2
        ns = 30
    else:
        raise ValueError("Test not implemented")

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=ft,
        n_shooting=ns,
        ode_solver=ode_solver(),
        assume_phase_dynamics=assume_phase_dynamics,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (ns * 20, 1))
        np.testing.assert_almost_equal(g, np.zeros((ns * 20, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (ns * 4, 1))
        np.testing.assert_almost_equal(g, np.zeros((ns * 4, 1)), decimal=6)

    # Check some results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]
    tf = sol.parameters["time"][0, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    if ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 0.2855606738489079)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((87.13363409, 0)), decimal=6)
        np.testing.assert_almost_equal(tau[:, -2], np.array((-99.99938226, 0)), decimal=6)

        # optimized time
        np.testing.assert_almost_equal(tf, 0.2855606738489079)

    elif ode_solver == OdeSolver.COLLOCATION:
        pass

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 0.2862324498580764)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((70.46234418, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-99.99964325, 0)))

        # optimized time
        np.testing.assert_almost_equal(tf, 0.2862324498580764)
    else:
        raise ValueError("Test not implemented")

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol, decimal_value=5)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
# @pytest.mark.parametrize("ode_solver", [OdeSolver.COLLOCATION])
def test_pendulum_min_time_mayer_constrained(ode_solver, assume_phase_dynamics):
    if platform.system() != "Linux":
        # This is a long test and CI is already long for Windows and Mac
        return

    # Load pendulum_min_time_Mayer
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Mayer as ocp_module

    # For reducing time assume_phase_dynamics=False is skipped for redundant tests
    if not assume_phase_dynamics and ode_solver == OdeSolver.COLLOCATION:
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    if ode_solver == OdeSolver.IRK:
        ft = 2
        ns = 35
        min_ft = 0.5
    elif ode_solver == OdeSolver.COLLOCATION:
        ft = 2
        ns = 10
        min_ft = 0.5
    elif ode_solver == OdeSolver.RK4:
        ft = 2
        ns = 30
        min_ft = 0.5
    else:
        raise ValueError("Test not implemented")

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=ft,
        n_shooting=ns,
        ode_solver=ode_solver(),
        min_time=min_ft,
        assume_phase_dynamics=assume_phase_dynamics,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (ns * 20, 1))
        np.testing.assert_almost_equal(g, np.zeros((ns * 20, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (ns * 4, 1))
        np.testing.assert_almost_equal(g, np.zeros((ns * 4, 1)), decimal=6)

    # Check some results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]
    tf = sol.parameters["time"][0, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 1.1878186850775596)
    else:
        np.testing.assert_almost_equal(f[0, 0], min_ft)

    # optimized time
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(tf, 1.1878186850775596)
    else:
        np.testing.assert_almost_equal(tf, min_ft)

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol, decimal_value=6)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_pendulum_max_time_mayer_constrained(ode_solver, assume_phase_dynamics):
    # Load pendulum_min_time_Mayer
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Mayer as ocp_module

    if platform.system() == "Windows" and not ode_solver != OdeSolver.RK4:
        # This is a long test and CI is already long for Windows
        return

    # For reducing time assume_phase_dynamics=False is skipped for redundant tests
    if not assume_phase_dynamics and ode_solver == OdeSolver.COLLOCATION:
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    if ode_solver == OdeSolver.IRK:
        ft = 2
        ns = 35
        max_ft = 1
    elif ode_solver == OdeSolver.COLLOCATION:
        ft = 2
        ns = 15
        max_ft = 1
    elif ode_solver == OdeSolver.RK4:
        ft = 2
        ns = 30
        max_ft = 1
    else:
        raise ValueError("Test not implemented")

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=ft,
        n_shooting=ns,
        ode_solver=ode_solver(),
        max_time=max_ft,
        weight=-1,
        assume_phase_dynamics=assume_phase_dynamics,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (ns * 20, 1))
        np.testing.assert_almost_equal(g, np.zeros((ns * 20, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (ns * 4, 1))
        np.testing.assert_almost_equal(g, np.zeros((ns * 4, 1)), decimal=6)

    # Check some results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]
    tf = sol.parameters["time"][0, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -1)

    np.testing.assert_almost_equal(tau[1, 0], np.array(0))
    np.testing.assert_almost_equal(tau[1, -2], np.array(0))

    # optimized time
    np.testing.assert_almost_equal(tf, max_ft)

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol, decimal_value=6)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_pendulum_min_time_lagrange(ode_solver, assume_phase_dynamics):
    # Load pendulum_min_time_Lagrange
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Lagrange as ocp_module

    if platform.system() == "Windows" and not assume_phase_dynamics:
        # This tst fails on the CI
        return

    # For reducing time assume_phase_dynamics=False is skipped for redundant tests
    if not assume_phase_dynamics and ode_solver == OdeSolver.COLLOCATION:
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    if ode_solver == OdeSolver.IRK:
        ft = 2
        ns = 35
    elif ode_solver == OdeSolver.COLLOCATION:
        ft = 2
        ns = 15
    elif ode_solver == OdeSolver.RK4:
        ft = 2
        ns = 42
    else:
        raise ValueError("Test not implemented")

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=ft,
        n_shooting=ns,
        ode_solver=ode_solver(),
        assume_phase_dynamics=assume_phase_dynamics,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (ns * 20, 1))
        np.testing.assert_almost_equal(g, np.zeros((ns * 20, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (ns * 4, 1))
        np.testing.assert_almost_equal(g, np.zeros((ns * 4, 1)), decimal=6)

    # Check some results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]
    tf = sol.parameters["time"][0, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    if ode_solver == OdeSolver.IRK:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.2855606738489078)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((87.13363409, 0)), decimal=6)
        np.testing.assert_almost_equal(tau[:, -2], np.array((-99.99938226, 0)), decimal=6)

        # optimized time
        np.testing.assert_almost_equal(tf, 0.2855606738489078)

    elif ode_solver == OdeSolver.COLLOCATION:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.8905637018911737)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((19.92168227, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-11.96503358, 0)))

        # optimized time
        np.testing.assert_almost_equal(tf, 0.8905637018911734)

    elif ode_solver == OdeSolver.RK4:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.28519514602152585)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((99.99914849, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-99.9990543, 0)))

        # optimized time
        np.testing.assert_almost_equal(tf, 0.28519514602152585)
    else:
        raise ValueError("Test not implemented")

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol, decimal_value=5)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_pendulum_min_time_lagrange_constrained(ode_solver):
    # Load pendulum_min_time_Lagrange
    biorbd_model_path = (TestUtils.bioptim_folder() + "/examples/optimal_time_ocp/models/pendulum.bioMod",)

    # --- Options --- #
    bio_model = BiorbdModel(biorbd_model_path[0])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME, min_bound=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    # ------------- #

    x_init = InitialGuessList()
    x_init.add([0] * (bio_model.nb_q + bio_model.nb_qdot))
    u_init = InitialGuessList()
    u_init.add([0] * bio_model.nb_tau)
    with pytest.raises(TypeError, match=re.escape("minimize_time() got an unexpected keyword argument 'min_bound'")):
        OptimalControlProgram(
            bio_model,
            dynamics,
            10,
            2,
            objective_functions=objective_functions,
            x_init=x_init,
            u_init=u_init,
        )


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_pendulum_max_time_lagrange_constrained(ode_solver):
    # Load pendulum_min_time_Lagrange
    biorbd_model_path = (TestUtils.bioptim_folder() + "/examples/optimal_time_ocp/models/pendulum.bioMod",)

    # --- Options --- #
    bio_model = BiorbdModel(biorbd_model_path[0])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME, weight=-1, max_bound=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    # ------------- #

    x_init = InitialGuessList()
    x_init.add([0] * (bio_model.nb_q + bio_model.nb_qdot))
    u_init = InitialGuessList()
    u_init.add([0] * bio_model.nb_tau)
    with pytest.raises(TypeError, match=re.escape("minimize_time() got an unexpected keyword argument 'max_bound'")):
        OptimalControlProgram(
            bio_model, dynamics, 10, 2, objective_functions=objective_functions, x_init=x_init, u_init=u_init
        )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_time_constraint(ode_solver, assume_phase_dynamics):
    if platform.system() != "Linux":
        # This is a long test and CI is already long for Windows and Mac
        return

    # Load time_constraint
    from bioptim.examples.optimal_time_ocp import time_constraint as ocp_module

    # For reducing time assume_phase_dynamics=False is skipped for redundant tests
    if not assume_phase_dynamics and ode_solver == OdeSolver.COLLOCATION:
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    if ode_solver == OdeSolver.IRK:
        ft = 2
        ns = 35
    elif ode_solver == OdeSolver.COLLOCATION:
        ft = 2
        ns = 15
    elif ode_solver == OdeSolver.RK4:
        ft = 2
        ns = 42
    else:
        raise ValueError("Test not implemented")

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=ft,
        n_shooting=ns,
        time_min=0.2,
        time_max=1,
        ode_solver=ode_solver(),
        assume_phase_dynamics=assume_phase_dynamics,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (ns * 20 + 1, 1))
        np.testing.assert_almost_equal(g, np.concatenate((np.zeros((ns * 20, 1)), [[1]])))
    else:
        np.testing.assert_equal(g.shape, (ns * 4 + 1, 1))
        np.testing.assert_almost_equal(g, np.concatenate((np.zeros((ns * 4, 1)), [[1]])))

    # Check some results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]
    tf = sol.parameters["time"][0, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # optimized time
    np.testing.assert_almost_equal(tf, 1.0)

    if ode_solver == OdeSolver.IRK:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 57.84641870505798)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((5.33802896, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-23.69200381, 0)))

    elif ode_solver == OdeSolver.COLLOCATION:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 90.22986699069487)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((8.48542163, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-18.13750096, 0)))

    elif ode_solver == OdeSolver.RK4:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 39.593354247030085)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((6.28713595, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-12.72892599, 0)))
    else:
        raise ValueError("Test not ready")

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol, decimal_value=6)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_monophase_time_constraint(ode_solver, assume_phase_dynamics):
    # Load time_constraint
    from bioptim.examples.optimal_time_ocp import multiphase_time_constraint as ocp_module

    # For reducing time assume_phase_dynamics=False is skipped for redundant tests
    if not assume_phase_dynamics and ode_solver == OdeSolver.RK8:
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        final_time=(2, 5, 4),
        time_min=(1, 3, 0.1),
        time_max=(2, 4, 0.8),
        n_shooting=(20,),
        ode_solver=ode_solver(),
        assume_phase_dynamics=assume_phase_dynamics,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 10826.61745902614)

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (120 * 5 + 7, 1))
        np.testing.assert_almost_equal(g, np.concatenate((np.zeros((120 * 5, 1)), np.array([[0, 0, 0, 0, 0, 0, 1]]).T)))
    else:
        np.testing.assert_equal(g.shape, (127, 1))
        np.testing.assert_almost_equal(g, np.concatenate((np.zeros((126, 1)), [[1]])))

    # Check some results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]
    tf = sol.parameters["time"][0, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 0)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((5.71428583, 9.81, 0)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((-5.71428583, 9.81, 0)))

    # optimized time
    np.testing.assert_almost_equal(tf, 1.0)

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_multiphase_time_constraint(ode_solver, assume_phase_dynamics):
    # Load time_constraint
    from bioptim.examples.optimal_time_ocp import multiphase_time_constraint as ocp_module

    # For reducing time assume_phase_dynamics=False is skipped for redundant tests
    if not assume_phase_dynamics and ode_solver == OdeSolver.COLLOCATION:
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        final_time=(2, 5, 4),
        time_min=(1, 3, 0.1),
        time_max=(2, 4, 0.8),
        n_shooting=(20, 30, 20),
        ode_solver=ode_solver(),
        assume_phase_dynamics=assume_phase_dynamics,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 55582.04125083612)

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (421 * 5 + 22, 1))
        np.testing.assert_almost_equal(
            g, np.concatenate((np.zeros((612, 1)), [[1]], np.zeros((909, 1)), [[3]], np.zeros((603, 1)), [[0.8]]))
        )
    else:
        np.testing.assert_equal(g.shape, (447, 1))
        np.testing.assert_almost_equal(
            g, np.concatenate((np.zeros((132, 1)), [[1]], np.zeros((189, 1)), [[3]], np.zeros((123, 1)), [[0.8]]))
        )

    # Check some results
    sol_merged = sol.merge_phases()
    states, controls = sol_merged.states, sol_merged.controls
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]
    tf_all = sol.parameters["time"]
    tf = sol_merged.phase_time[1]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((5.71428583, 9.81, 0)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((-8.92857121, 9.81, -14.01785679)))

    # optimized time
    np.testing.assert_almost_equal(tf_all.T, [[1.0, 3, 0.8]])
    np.testing.assert_almost_equal(tf, np.sum(tf_all))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def partial_ocp_parameters(n_phases):
    if n_phases != 1 and n_phases != 3:
        raise RuntimeError("n_phases should be 1 or 3")

    biorbd_model_path = TestUtils.bioptim_folder() + "/examples/optimal_time_ocp/models/cube.bioMod"
    bio_model = BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path)
    n_shooting = (2, 2, 2)
    final_time = (2, 5, 4)
    time_min = [1, 3, 0.1]
    time_max = [2, 4, 0.8]
    tau_min, tau_max, tau_init = -100, 100, 0
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    if n_phases > 1:
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    if n_phases > 1:
        x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
        x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    for bounds in x_bounds:
        for i in [1, 3, 4, 5]:
            bounds.min[i, [0, -1]] = 0
            bounds.max[i, [0, -1]] = 0
    x_bounds[0].min[2, 0] = 0.0
    x_bounds[0].max[2, 0] = 0.0
    if n_phases > 1:
        x_bounds[2].min[2, [0, -1]] = [0.0, 1.57]
        x_bounds[2].max[2, [0, -1]] = [0.0, 1.57]

    x_init = InitialGuessList()
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
    if n_phases > 1:
        x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
        x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))

    u_bounds = BoundsList()
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)
    if n_phases > 1:
        u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)
        u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model[0].nb_tau)
    if n_phases > 1:
        u_init.add([tau_init] * bio_model[0].nb_tau)
        u_init.add([tau_init] * bio_model[0].nb_tau)

    return (
        bio_model[:n_phases],
        n_shooting[:n_phases],
        final_time[:n_phases],
        time_min[:n_phases],
        time_max[:n_phases],
        tau_min,
        tau_max,
        tau_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    )


def test_mayer_neg_monophase_time_constraint():
    (
        bio_model,
        n_shooting,
        final_time,
        time_min,
        time_max,
        torque_min,
        torque_max,
        torque_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    ) = partial_ocp_parameters(n_phases=1)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, minimum=time_min[0], maximum=time_max[0])

    with pytest.raises(RuntimeError, match="Time constraint/objective cannot declare more than once"):
        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
            constraints,
        )


def test_mayer1_neg_multiphase_time_constraint():
    (
        bio_model,
        n_shooting,
        final_time,
        time_min,
        time_max,
        torque_min,
        torque_max,
        torque_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    ) = partial_ocp_parameters(n_phases=3)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=0)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, minimum=time_min[0], maximum=time_max[0])

    with pytest.raises(RuntimeError, match="Time constraint/objective cannot declare more than once"):
        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
            constraints,
        )


def test_mayer2_neg_multiphase_time_constraint():
    (
        bio_model,
        n_shooting,
        final_time,
        time_min,
        time_max,
        torque_min,
        torque_max,
        torque_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    ) = partial_ocp_parameters(n_phases=3)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=2)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=2)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, minimum=time_min[0], maximum=time_max[0], phase=2)

    with pytest.raises(RuntimeError, match="Time constraint/objective cannot declare more than once"):
        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
            constraints,
        )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_mayer_multiphase_time_constraint(assume_phase_dynamics):
    (
        bio_model,
        n_shooting,
        final_time,
        time_min,
        time_max,
        torque_min,
        torque_max,
        torque_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    ) = partial_ocp_parameters(n_phases=3)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=0)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=2)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, minimum=time_min[0], maximum=time_max[0], phase=2)

    OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_lagrange_neg_monophase_time_constraint(assume_phase_dynamics):
    (
        bio_model,
        n_shooting,
        final_time,
        time_min,
        time_max,
        torque_min,
        torque_max,
        torque_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    ) = partial_ocp_parameters(n_phases=1)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, minimum=time_min[0], maximum=time_max[0])

    with pytest.raises(RuntimeError, match="Time constraint/objective cannot declare more than once"):
        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
            constraints,
            assume_phase_dynamics=assume_phase_dynamics,
        )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_lagrange1_neg_multiphase_time_constraint(assume_phase_dynamics):
    with pytest.raises(RuntimeError, match="Time constraint/objective cannot declare more than once"):
        (
            bio_model,
            n_shooting,
            final_time,
            time_min,
            time_max,
            torque_min,
            torque_max,
            torque_init,
            dynamics,
            x_bounds,
            x_init,
            u_bounds,
            u_init,
        ) = partial_ocp_parameters(n_phases=3)

        objective_functions = ObjectiveList()
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME, phase=0)

        constraints = ConstraintList()
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=0
        )
        constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, minimum=time_min[0], maximum=time_max[0], phase=0)

        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
            constraints,
            assume_phase_dynamics=assume_phase_dynamics,
        )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_lagrange2_neg_multiphase_time_constraint(assume_phase_dynamics):
    with pytest.raises(RuntimeError, match="Time constraint/objective cannot declare more than once"):
        (
            bio_model,
            n_shooting,
            final_time,
            time_min,
            time_max,
            torque_min,
            torque_max,
            torque_init,
            dynamics,
            x_bounds,
            x_init,
            u_bounds,
            u_init,
        ) = partial_ocp_parameters(n_phases=3)

        objective_functions = ObjectiveList()
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME, phase=2)

        constraints = ConstraintList()
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=2
        )
        constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, minimum=time_min[0], maximum=time_max[0], phase=2)

        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
            constraints,
            assume_phase_dynamics=assume_phase_dynamics,
        )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_lagrange_multiphase_time_constraint(assume_phase_dynamics):
    (
        bio_model,
        n_shooting,
        final_time,
        time_min,
        time_max,
        torque_min,
        torque_max,
        torque_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    ) = partial_ocp_parameters(n_phases=3)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME, phase=0)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=2)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, minimum=time_min[0], maximum=time_max[0], phase=2)

    OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_mayer_neg_two_objectives(assume_phase_dynamics):
    (
        bio_model,
        n_shooting,
        final_time,
        time_min,
        time_max,
        torque_min,
        torque_max,
        torque_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    ) = partial_ocp_parameters(n_phases=1)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=0)

    with pytest.raises(RuntimeError, match="Time constraint/objective cannot declare more than once"):
        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
            assume_phase_dynamics=assume_phase_dynamics,
        )
