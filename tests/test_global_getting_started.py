"""
Test for file IO
"""
import pickle
from pickle import PicklingError
import re

import pytest
import numpy as np
from bioptim import InterpolationType, OdeSolver

from .utils import TestUtils


def test_pendulum_save_and_load():
    bioptim_folder = TestUtils.bioptim_folder()
    pendulum = TestUtils.load_module(bioptim_folder + "/examples/getting_started/pendulum.py")

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod",
        final_time=2,
        n_shooting=10,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 6657.974502951726)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((16.25734477, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-25.59944635, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("n_threads", [1, 2])
@pytest.mark.parametrize("use_sx", [False, True])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_pendulum_save_and_load(n_threads, use_sx, ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    pendulum = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_save_and_load.py")
    ode_solver = ode_solver()

    if isinstance(ode_solver, OdeSolver.IRK):
        if use_sx:
            with pytest.raises(NotImplementedError, match="use_sx=True and OdeSolver.IRK are not yet compatible"):
                pendulum.prepare_ocp(
                    biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod",
                    final_time=2,
                    n_shooting=10,
                    n_threads=n_threads,
                    use_sx=use_sx,
                    ode_solver=ode_solver,
                )
        else:
            ocp = pendulum.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod",
                final_time=2,
                n_shooting=10,
                n_threads=n_threads,
                use_sx=use_sx,
                ode_solver=ode_solver,
            )
            sol = ocp.solve()

            # Check objective function value
            f = np.array(sol.cost)
            np.testing.assert_equal(f.shape, (1, 1))
            np.testing.assert_almost_equal(f[0, 0], 6644.75968052)

            # Check constraints
            g = np.array(sol.constraints)
            np.testing.assert_equal(g.shape, (40, 1))
            np.testing.assert_almost_equal(g, np.zeros((40, 1)))

            # Check some of the results
            q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
            np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

            # initial and final velocities
            np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
            np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

            # initial and final controls
            np.testing.assert_almost_equal(tau[:, 0], np.array((16.23831574, 0)))
            np.testing.assert_almost_equal(tau[:, -1], np.array((-25.59884582, 0)))

            # save and load
            TestUtils.save_and_load(sol, ocp, True)

            # simulate
            TestUtils.simulate(sol)
    else:
        ocp = pendulum.prepare_ocp(
            biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod",
            final_time=2,
            n_shooting=10,
            n_threads=n_threads,
            use_sx=use_sx,
            ode_solver=ode_solver,
        )
        sol = ocp.solve()

        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        if isinstance(ode_solver, OdeSolver.RK8):
            np.testing.assert_almost_equal(f[0, 0], 6654.69715318338)
        else:
            np.testing.assert_almost_equal(f[0, 0], 6657.974502951726)

        # Check constraints
        g = np.array(sol.constraints)
        np.testing.assert_equal(g.shape, (40, 1))
        np.testing.assert_almost_equal(g, np.zeros((40, 1)))

        # Check some of the results
        q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

        # initial and final controls
        if isinstance(ode_solver, OdeSolver.RK8):
            np.testing.assert_almost_equal(tau[:, 0], np.array((16.2560473, 0)))
            np.testing.assert_almost_equal(tau[:, -1], np.array((-25.5991168, 0)))
        else:
            np.testing.assert_almost_equal(tau[:, 0], np.array((16.25734477, 0)))
            np.testing.assert_almost_equal(tau[:, -1], np.array((-25.59944635, 0)))

        # save and load
        TestUtils.save_and_load(sol, ocp, True)

        # simulate
        TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_custom_constraint_track_markers(ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    custom_constraint = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_constraint.py")
    ode_solver = ode_solver()

    ocp = custom_constraint.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod", ode_solver=ode_solver
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (186, 1))
    np.testing.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))

    if isinstance(ode_solver, OdeSolver.IRK):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 19767.53312569523)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((1.4516129, 9.81, 2.27903226)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-1.45161291, 9.81, -2.27903226)))
    else:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 19767.533125695223)

        np.testing.assert_almost_equal(tau[:, 0], np.array((1.4516128810214546, 9.81, 2.2790322540381487)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-1.4516128810214546, 9.81, -2.2790322540381487)))


@pytest.mark.parametrize("interpolation", InterpolationType)
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_initial_guesses(interpolation, ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    initial_guess = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_initial_guess.py")
    ode_solver = ode_solver()

    np.random.seed(42)
    ocp = initial_guess.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
        final_time=1,
        n_shooting=5,
        initial_guess=interpolation,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 13954.735)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (36, 1))
    np.testing.assert_almost_equal(g, np.zeros((36, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([1, 0, 0]))
    np.testing.assert_almost_equal(q[:, -1], np.array([2, 0, 1.57]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([5.0, 9.81, 7.85]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-5.0, 9.81, -7.85]))

    # save and load
    if interpolation == InterpolationType.CUSTOM:
        with pytest.raises(AttributeError, match="'PathCondition' object has no attribute 'custom_function'"):
            TestUtils.save_and_load(sol, ocp, True)
    else:
        TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_cyclic_objective(ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    cyclic_movement = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_cyclic_movement.py")
    ode_solver = ode_solver()

    np.random.seed(42)
    ocp = cyclic_movement.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
        final_time=1,
        n_shooting=10,
        loop_from_constraint=False,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 56851.88181545)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (67, 1))
    np.testing.assert_almost_equal(g, np.zeros((67, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([1.60205103, -0.01069317, 0.62477988]))
    np.testing.assert_almost_equal(q[:, -1], np.array([1, 0, 1.57]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0.12902365, 0.09340155, -0.20256713)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([9.89210954, 9.39362112, -15.53061197]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([17.16370432, 9.78643138, -26.94701577]))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_cyclic_constraint(ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    cyclic_movement = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_cyclic_movement.py")
    ode_solver = ode_solver()

    np.random.seed(42)
    ocp = cyclic_movement.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
        final_time=1,
        n_shooting=10,
        loop_from_constraint=True,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 78921.61000000016)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (73, 1))
    np.testing.assert_almost_equal(g, np.zeros((73, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([1, 0, 1.57]))
    np.testing.assert_almost_equal(q[:, -1], np.array([1, 0, 1.57]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([20.0, 9.81, -31.4]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([20.0, 9.81, -31.4]))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_phase_transitions(ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    phase_transition = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_phase_transitions.py")
    ode_solver = ode_solver()

    ocp = phase_transition.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod", ode_solver=ode_solver
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 110875.0772043361)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (515, 1))
    np.testing.assert_almost_equal(g, np.zeros((515, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls

    # initial and final position
    np.testing.assert_almost_equal(states[0]["q"][:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(states[-1]["q"][:, -1], np.array((1, 0, 0)))
    # initial and final velocities
    np.testing.assert_almost_equal(states[0]["qdot"][:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(states[-1]["qdot"][:, -1], np.array((0, 0, 0)))

    # cyclic continuity (between phase 3 and phase 0)
    np.testing.assert_almost_equal(states[-1]["q"][:, -1], states[0]["q"][:, 0])

    # Continuity between phase 0 and phase 1
    np.testing.assert_almost_equal(states[0]["q"][:, -1], states[1]["q"][:, 0])

    if isinstance(ode_solver, OdeSolver.IRK):
        # initial and final controls
        np.testing.assert_almost_equal(controls[0]["tau"][:, 0], np.array((0.95986719, 9.70855983, -0.06237331)))
        np.testing.assert_almost_equal(controls[-1]["tau"][:, -2], np.array((0, 1.27170519e01, 1.14878049e00)))
    else:
        # initial and final controls
        np.testing.assert_almost_equal(controls[0]["tau"][:, 0], np.array((0.9598672, 9.7085598, -0.0623733)))
        np.testing.assert_almost_equal(controls[-1]["tau"][:, -2], np.array((0, 1.2717052e01, 1.1487805e00)))

    # save and load
    with pytest.raises(PicklingError, match="import of module 'custom_phase_transitions' failed"):
        TestUtils.save_and_load(sol, ocp, True)

    # simulate
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Phase transition must have the same number of states (2) "
            "when integrating with Shooting.SINGLE_CONTINUOUS. If it is not possible, "
            "please integrate with Shooting.SINGLE"
        ),
    ):
        TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_parameter_optimization(ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    parameter = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_parameters.py")
    ode_solver = ode_solver()

    ocp = parameter.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        optim_gravity=True,
        optim_mass=False,
        min_g=np.array([-1, -1, -10]),
        max_g=np.array([1, 1, -5]),
        min_m=10,
        max_m=30,
        target_g=np.array([0, 0, -9.81]),
        target_m=20,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (400, 1))
    np.testing.assert_almost_equal(g, np.zeros((400, 1)))

    # Check some of the results
    q, qdot, tau, gravity = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.parameters["gravity_xyz"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    if isinstance(ode_solver, OdeSolver.IRK):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 801.7834735768548, decimal=6)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((6.3025795, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-8.8472355, 0)))

        # gravity parameter
        np.testing.assert_almost_equal(gravity, np.array([[0, 0.0902555, -9.7896801]]).T)

    elif isinstance(ode_solver, OdeSolver.RK8):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 801.7834735768548, decimal=6)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((6.3025795, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-8.8472355, 0)))

        # gravity parameter
        np.testing.assert_almost_equal(gravity, np.array([[0, 0.0902555, -9.7896801]]).T)

    else:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 801.7834735768548, decimal=6)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((6.3025795, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-8.8472355, 0)))

        # gravity parameter
        np.testing.assert_almost_equal(gravity, np.array([[0, 0.0902555, -9.7896801]]).T)

    # save and load
    with pytest.raises(PicklingError, match="import of module 'custom_parameters' failed"):
        TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("problem_type_custom", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_custom_problem_type_and_dynamics(problem_type_custom, ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    pendulum = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_dynamics.py")
    ode_solver = ode_solver()

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
        problem_type_custom=problem_type_custom,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 19767.5331257)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (186, 1))
    np.testing.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.4516129, 9.81, 2.27903226)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((-1.45161291, 9.81, -2.27903226)))


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_example_external_forces(ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    external_forces = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_external_forces.py")
    ode_solver = ode_solver()

    ocp = external_forces.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube_with_forces.bioMod",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 9875.88768746912)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (246, 1))
    np.testing.assert_almost_equal(g, np.zeros((246, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0, 9.71322593, 0, 0)))
    np.testing.assert_almost_equal(tau[:, 10], np.array((0, 7.71100122, 0, 0)))
    np.testing.assert_almost_equal(tau[:, 20], np.array((0, 5.70877651, 0, 0)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0, 3.90677425, 0, 0)))

    if isinstance(ode_solver, OdeSolver.IRK):

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0, 0, 0, 0)), decimal=5)
        np.testing.assert_almost_equal(q[:, -1], np.array((0, 2, 0, 0)), decimal=5)

        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)), decimal=5)
        np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)), decimal=5)
    else:

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0, 0, 0, 0)))
        np.testing.assert_almost_equal(q[:, -1], np.array((0, 2, 0, 0)))

        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
        np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_example_multiphase(ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    multiphase = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_multiphase.py")
    ode_solver = ode_solver()

    ocp = multiphase.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod", ode_solver=ode_solver
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 106088.01707867868)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (444, 1))
    np.testing.assert_almost_equal(g, np.zeros((444, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls

    # initial and final position
    np.testing.assert_almost_equal(states[0]["q"][:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(states[0]["q"][:, -1], np.array((2, 0, 0.0078695)))
    np.testing.assert_almost_equal(states[1]["q"][:, 0], np.array((2, 0, 0.0078695)))
    np.testing.assert_almost_equal(states[1]["q"][:, -1], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(states[2]["q"][:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(states[2]["q"][:, -1], np.array((2, 0, 1.57)))

    # initial and final velocities
    np.testing.assert_almost_equal(states[0]["qdot"][:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(states[0]["qdot"][:, -1], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(states[1]["qdot"][:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(states[1]["qdot"][:, -1], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(states[2]["qdot"][:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(states[2]["qdot"][:, -1], np.array((0, 0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(controls[0]["tau"][:, 0], np.array((1.42857142, 9.81, 0.01124212)))
    np.testing.assert_almost_equal(controls[0]["tau"][:, -1], np.array((-1.42857144, 9.81, -0.01124212)))
    np.testing.assert_almost_equal(controls[1]["tau"][:, 0], np.array((-0.22788183, 9.81, 0.01775688)))
    np.testing.assert_almost_equal(controls[1]["tau"][:, -1], np.array((0.2957136, 9.81, 0.285805)))
    np.testing.assert_almost_equal(controls[2]["tau"][:, 0], np.array((0.3078264, 9.81, 0.34001243)))
    np.testing.assert_almost_equal(controls[2]["tau"][:, -1], np.array((-0.36233407, 9.81, -0.58394606)))

    # save and load
    with pytest.raises(pickle.PickleError):
        TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_contact_forces_inequality_GREATER_THAN_constraint(ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    contact = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_inequality_constraint.py")
    min_bound = 50
    ode_solver = ode_solver()

    ocp = contact.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/2segments_4dof_2contacts.bioMod",
        phase_time=0.3,
        n_shooting=10,
        min_bound=min_bound,
        max_bound=np.inf,
        mu=0.2,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.15132909609835643)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (120, 1))
    np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
    np.testing.assert_array_less(-g[80:100], -min_bound)

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.75, 0.75)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-0.2142318, 0.11786579, -0.25596094, 0.25596094)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-1.28034999, 0.3350692, 2.64693595, -2.64693595)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-33.50426941)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-15.61654842)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_contact_forces_inequality_LESSER_THAN_constraint(ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    contact = TestUtils.load_module(bioptim_folder + "/examples/getting_started/example_inequality_constraint.py")
    max_bound = 75
    ode_solver = ode_solver()

    ocp = contact.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/2segments_4dof_2contacts.bioMod",
        phase_time=0.3,
        n_shooting=10,
        min_bound=-np.inf,
        max_bound=max_bound,
        mu=0.2,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.16913696624413754)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (120, 1))
    np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
    np.testing.assert_array_less(g[80:100], max_bound)

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.75, 0.75)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-0.10473449, 0.07490939, -0.4917506, 0.4917506)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-0.89054233, 0.47700932, 2.02049847, -2.02049847)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-24.36641199)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-23.53987687)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)
