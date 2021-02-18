"""
Test for file IO
"""
import importlib.util
from pickle import PicklingError
from pathlib import Path
import re

import pytest
import numpy as np
from bioptim import InterpolationType, OdeSolver

from .utils import TestUtils


def test_pendulum_save_and_load():
    # Load pendulum
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "pendulum", str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.py"
    )
    pendulum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pendulum)

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
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
    # Load pendulum
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "pendulum", str(PROJECT_FOLDER) + "/examples/getting_started/example_save_and_load.py"
    )
    pendulum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pendulum)

    if ode_solver == OdeSolver.IRK:
        if use_sx:
            with pytest.raises(NotImplementedError, match="use_sx and OdeSolver.IRK are not yet compatible"):
                pendulum.prepare_ocp(
                    biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
                    final_time=2,
                    n_shooting=10,
                    n_threads=n_threads,
                    use_sx=use_sx,
                    ode_solver=ode_solver,
                )
        else:
            ocp = pendulum.prepare_ocp(
                biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
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
            biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
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
        if ode_solver == OdeSolver.RK8:
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
        if ode_solver == OdeSolver.RK8:
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
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "custom_constraint", str(PROJECT_FOLDER) + "/examples/getting_started/custom_constraint.py"
    )
    custom_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_constraint)

    ocp = custom_constraint.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod", ode_solver=ode_solver
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

    if ode_solver == OdeSolver.IRK:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 19767.53312569523)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((1.4516129, 9.81, 2.27903226)))
        np.testing.assert_almost_equal(tau[:, -1], np.array((-1.45161291, 9.81, -2.27903226)))
    else:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 19767.533125695223)

        np.testing.assert_almost_equal(tau[:, 0], np.array((1.4516128810214546, 9.81, 2.2790322540381487)))
        np.testing.assert_almost_equal(tau[:, -1], np.array((-1.4516128810214546, 9.81, -2.2790322540381487)))


@pytest.mark.parametrize("interpolation", InterpolationType)
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_initial_guesses(interpolation, ode_solver):
    #  Load initial_guess
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "initial_guess", str(PROJECT_FOLDER) + "/examples/getting_started/custom_initial_guess.py"
    )
    initial_guess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(initial_guess)

    np.random.seed(42)
    ocp = initial_guess.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod",
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
    np.testing.assert_almost_equal(tau[:, -1], np.array([-5.0, 9.81, -7.85]))

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
    #  Load initial_guess
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "initial_guess", str(PROJECT_FOLDER) + "/examples/getting_started/example_cyclic_movement.py"
    )
    cyclic_movement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cyclic_movement)

    np.random.seed(42)
    ocp = cyclic_movement.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod",
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
    np.testing.assert_almost_equal(tau[:, -1], np.array([17.16370432, 9.78643138, -26.94701577]))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_cyclic_constraint(ode_solver):
    #  Load initial_guess
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "initial_guess", str(PROJECT_FOLDER) + "/examples/getting_started/example_cyclic_movement.py"
    )
    cyclic_movement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cyclic_movement)

    np.random.seed(42)
    ocp = cyclic_movement.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod",
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
    np.testing.assert_almost_equal(tau[:, -1], np.array([20.0, 9.81, -31.4]))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_phase_transitions(ode_solver):
    # Load phase_transitions
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "phase_transitions", str(PROJECT_FOLDER) + "/examples/getting_started/custom_phase_transitions.py"
    )
    phase_transition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase_transition)

    ocp = phase_transition.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod", ode_solver=ode_solver
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

    if ode_solver == OdeSolver.IRK:
        # initial and final controls
        np.testing.assert_almost_equal(controls[0]["tau"][:, 0], np.array((0.95986719, 9.70855983, -0.06237331)))
        np.testing.assert_almost_equal(controls[-1]["tau"][:, -1], np.array((0, 1.27170519e01, 1.14878049e00)))
    else:
        # initial and final controls
        np.testing.assert_almost_equal(controls[0]["tau"][:, 0], np.array((0.9598672, 9.7085598, -0.0623733)))
        np.testing.assert_almost_equal(controls[-1]["tau"][:, -1], np.array((0, 1.2717052e01, 1.1487805e00)))

    # save and load
    with pytest.raises(PicklingError, match="import of module 'phase_transitions' failed"):
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
    # Load phase_transitions
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "parameter_optimization", str(PROJECT_FOLDER) + "/examples/getting_started/custom_parameters.py"
    )
    parameter_optimization = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parameter_optimization)

    ocp = parameter_optimization.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
        final_time=3,
        n_shooting=20,
        min_g=-10,
        max_g=-6,
        target_g=-8,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (80, 1))
    np.testing.assert_almost_equal(g, np.zeros((80, 1)))

    # Check some of the results
    q, qdot, tau, gravity = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.parameters["gravity_z"]

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
        np.testing.assert_almost_equal(f[0, 0], 853.5298104707485, decimal=6)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((8.13135278, 0)))
        np.testing.assert_almost_equal(tau[:, -1], np.array((-7.91821551, 0)))

        # gravity parameter
        np.testing.assert_almost_equal(gravity, np.array([[-9.0988827]]))

    elif ode_solver == OdeSolver.RK8:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 853.5348080507781, decimal=6)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((8.1317708, 0)))
        np.testing.assert_almost_equal(tau[:, -1], np.array((-7.9180414, 0)))

        # gravity parameter
        np.testing.assert_almost_equal(gravity, np.array([[-9.0988849]]))

    else:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 853.5406085230834, decimal=6)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((8.1318336, 0)))
        np.testing.assert_almost_equal(tau[:, -1], np.array((-7.91806351, 0)))

        # gravity parameter
        np.testing.assert_almost_equal(gravity, np.array([[-9.09889371]]))

    # save and load
    with pytest.raises(PicklingError, match="import of module 'parameter_optimization' failed"):
        TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("problem_type_custom", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_custom_problem_type_and_dynamics(problem_type_custom, ode_solver):
    # Load pendulum
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "custom_problem_type_and_dynamics",
        str(PROJECT_FOLDER) + "/examples/getting_started/custom_dynamics.py",
    )
    pendulum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pendulum)

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod",
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
    np.testing.assert_almost_equal(tau[:, -1], np.array((-1.45161291, 9.81, -2.27903226)))


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_example_external_forces(ode_solver):
    # Load external_forces
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "external_forces", str(PROJECT_FOLDER) + "/examples/getting_started/example_external_forces.py"
    )
    external_forces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(external_forces)

    ocp = external_forces.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube_with_forces.bioMod",
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
    np.testing.assert_almost_equal(tau[:, -1], np.array((0, 3.90677425, 0, 0)))

    if ode_solver == OdeSolver.IRK:

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
    # Load multiphase_track_markers
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "multiphase_track_markers", str(PROJECT_FOLDER) + "/examples/getting_started/example_multiphase.py"
    )
    multiphase_track_markers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(multiphase_track_markers)

    ocp = multiphase_track_markers.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod", ode_solver=ode_solver
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 106084.82631762947)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (444, 1))
    np.testing.assert_almost_equal(g, np.zeros((444, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls

    # initial and final position
    np.testing.assert_almost_equal(states[0]["q"][:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(states[0]["q"][:, -1], np.array((2, 0, 0)))
    np.testing.assert_almost_equal(states[1]["q"][:, 0], np.array((2, 0, 0)))
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
    np.testing.assert_almost_equal(controls[0]["tau"][:, 0], np.array((1.42857142, 9.81, 0)))
    np.testing.assert_almost_equal(controls[0]["tau"][:, -1], np.array((-1.42857144, 9.81, 0)))
    np.testing.assert_almost_equal(controls[1]["tau"][:, 0], np.array((-0.2322581, 9.81, 0.0)))
    np.testing.assert_almost_equal(controls[1]["tau"][:, -1], np.array((0.2322581, 9.81, -0.0)))
    np.testing.assert_almost_equal(controls[2]["tau"][:, 0], np.array((0.35714285, 9.81, 0.56071428)))
    np.testing.assert_almost_equal(controls[2]["tau"][:, -1], np.array((-0.35714285, 9.81, -0.56071428)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_contact_forces_inequality_GREATER_THAN_constraint(ode_solver):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "contact_forces_inequality_constraint",
        str(PROJECT_FOLDER) + "/examples/getting_started/example_inequality_constraint.py",
    )
    contact_forces_inequality_GREATER_THAN_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contact_forces_inequality_GREATER_THAN_constraint)

    min_bound = 50
    ocp = contact_forces_inequality_GREATER_THAN_constraint.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/getting_started/2segments_4dof_2contacts.bioMod",
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
    np.testing.assert_array_less(-g[80:101], -min_bound)

    expected_pos_g = np.array(
        [
            [62.85371865],
            [62.82632694],
            [62.77061632],
            [62.80726884],
            [63.09997796],
            [69.3487235],
            [73.43121159],
            [74.7667476],
            [74.89375463],
            [73.47681824],
            [101.10797073],
            [99.62241641],
            [95.37642224],
            [89.06990638],
            [81.60740766],
            [57.80803068],
            [50.02360345],
            [50.01163094],
            [50.00758125],
            [50.00019548],
            [65.58087959],
            [3.79616622e-03],
            [6.49747810e01],
            [4.71631139e-03],
            [6.32525592e01],
            [6.25620077e-03],
            [6.07415162e01],
            [9.35385817e-03],
            [5.78642472e01],
            [1.87070469e-02],
            [3.74727918e01],
            [1.33899099e01],
            [3.34212028e01],
            [1.59607232e01],
            [3.81788678e01],
            [1.17324837e01],
            [4.35247132e01],
            [6.43582114e00],
            [4.93906118e01],
            [1.93682902e-04],
        ]
    )
    np.testing.assert_almost_equal(g[80:], expected_pos_g, decimal=5)

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.75, 0.75)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-0.2142318, 0.11786579, -0.25596094, 0.25596094)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-1.28034999, 0.3350692, 2.64693595, -2.64693595)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-33.50426976)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-15.61654842)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_contact_forces_inequality_LESSER_THAN_constraint(ode_solver):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "contact_forces_inequality_constraint",
        str(PROJECT_FOLDER) + "/examples/getting_started/example_inequality_constraint.py",
    )
    contact_forces_inequality_LESSER_THAN_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contact_forces_inequality_LESSER_THAN_constraint)

    max_bound = 75
    ocp = contact_forces_inequality_LESSER_THAN_constraint.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/getting_started/2segments_4dof_2contacts.bioMod",
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
    np.testing.assert_array_less(g[80:101], max_bound)
    expected_non_zero_g = np.array(
        [
            [68.19948996],
            [68.2365752],
            [68.34252145],
            [68.50038511],
            [68.67822587],
            [68.82285626],
            [68.84956991],
            [68.62603739],
            [67.94761301],
            [67.42619074],
            [74.99980706],
            [74.99977204],
            [74.9997287],
            [74.99967386],
            [74.9996024],
            [74.99950547],
            [74.99936655],
            [74.99915239],
            [74.99882431],
            [72.63824447],
            [39.14150624],
            [1.81382126e01],
            [3.93259055e01],
            [1.79686334e01],
            [3.98873239e01],
            [1.74495761e01],
            [4.08504997e01],
            [1.65495239e01],
            [4.22574762e01],
            [1.52136551e01],
            [4.41685227e01],
            [1.33604220e01],
            [4.66626670e01],
            [1.08769075e01],
            [4.98366511e01],
            [7.61342478e00],
            [5.38000010e01],
            [3.37857394e00],
            [5.60243588e01],
            [1.41530765e-03],
        ]
    )
    np.testing.assert_almost_equal(g[80:], expected_non_zero_g, decimal=5)

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.75, 0.75)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-0.10473449, 0.07490939, -0.4917506, 0.4917506)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-0.89054259, 0.47700939, 2.02049899, -2.02049899)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-24.36641247)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-23.53986556)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)
