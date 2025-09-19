"""
Test for file IO
"""

import platform

from bioptim import (
    TorqueBiorbdModel,
    ConstraintList,
    ConstraintFcn,
    DynamicsOptionsList,
    InitialGuessList,
    BoundsList,
    Node,
    ObjectiveList,
    ObjectiveFcn,
    OptimalControlProgram,
    OdeSolver,
    ControlType,
    PhaseDynamics,
    SolutionMerge,
    DynamicsOptions,
)
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK, OdeSolver.TRAPEZOIDAL])
def test_pendulum_max_time_mayer_constrained(ode_solver, phase_dynamics):
    # Load pendulum_min_time_Mayer
    from bioptim.examples.toy_examples.optimal_time_ocp import pendulum_min_time_Mayer as ocp_module

    if platform.system() == "Windows" and ode_solver == OdeSolver.COLLOCATION:
        pytest.skip("These tests do not pass on Windows.")

    bioptim_folder = TestUtils.bioptim_folder()

    ns = 30
    tf = 1
    max_tf = 0.5
    control_type = ControlType.CONSTANT_WITH_LAST_NODE if ode_solver == OdeSolver.TRAPEZOIDAL else ControlType.CONSTANT

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
        final_time=tf,
        n_shooting=ns,
        ode_solver=ode_solver(),
        max_time=max_tf,
        weight=-1,
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
        control_type=control_type,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        npt.assert_equal(g.shape, (ns * 20, 1))
        npt.assert_almost_equal(g, np.zeros((ns * 20, 1)), decimal=6)
    else:
        npt.assert_equal(g.shape, (ns * 4, 1))
        npt.assert_almost_equal(g, np.zeros((ns * 4, 1)), decimal=6)

    # Check some results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]
    tf = sol.decision_time(to_merge=SolutionMerge.NODES)[-1, 0]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((0, 0)))
    npt.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], -max_tf, decimal=5)

    npt.assert_almost_equal(tau[1, 0], np.array(0))
    npt.assert_almost_equal(tau[1, -1], np.array(0))

    # optimized time
    npt.assert_almost_equal(tf, max_tf, decimal=5)

    # simulate
    TestUtils.simulate(sol, decimal_value=5)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_time_constraint(ode_solver, phase_dynamics):

    if platform.system() == "Windows" and ode_solver == OdeSolver.COLLOCATION:
        pytest.skip("These tests are sensitive on Windows.")

    # Load time_constraint
    from bioptim.examples.toy_examples.optimal_time_ocp import time_constraint as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

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
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
        final_time=ft,
        n_shooting=ns,
        time_min=0.2,
        time_max=1,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        npt.assert_equal(g.shape, (ns * 20 + 1, 1))
        npt.assert_almost_equal(g, np.concatenate((np.zeros((ns * 20, 1)), [[1]])))
    else:
        npt.assert_equal(g.shape, (ns * 4 + 1, 1))
        npt.assert_almost_equal(g, np.concatenate((np.zeros((ns * 4, 1)), [[1]])))

    # Check some results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]
    tf = sol.decision_time(to_merge=SolutionMerge.NODES)[-1, 0]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((0, 0)))
    npt.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # optimized time
    npt.assert_almost_equal(tf, 1.0)

    if ode_solver == OdeSolver.IRK:
        # Check objective function value
        f = np.array(sol.cost)
        npt.assert_equal(f.shape, (1, 1))
        npt.assert_almost_equal(f[0, 0], 57.84641870505798)

        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array((5.33802896, 0)))
        npt.assert_almost_equal(tau[:, -1], np.array((-23.69200381, 0)))

    elif ode_solver == OdeSolver.COLLOCATION:
        # Check objective function value
        f = np.array(sol.cost)
        npt.assert_equal(f.shape, (1, 1))
        npt.assert_almost_equal(f[0, 0], 94.3161259540302)

        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array((10.47494692, 0)))
        npt.assert_almost_equal(tau[:, -1], np.array((-19.49344386, 0)))

    elif ode_solver == OdeSolver.RK4:
        # Check objective function value
        f = np.array(sol.cost)
        npt.assert_equal(f.shape, (1, 1))
        npt.assert_almost_equal(f[0, 0], 39.593354247030085)

        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array((6.28713595, 0)))
        npt.assert_almost_equal(tau[:, -1], np.array((-12.72892599, 0)))
    else:
        raise ValueError("Test not ready")

    # simulate
    TestUtils.simulate(sol, decimal_value=6)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_monophase_time_constraint(ode_solver, phase_dynamics):
    # Load time_constraint
    from bioptim.examples.toy_examples.optimal_time_ocp import multiphase_time_constraint as ocp_module

    # For reducing time phase_dynamics=PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.RK8:
        return

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube.bioMod",
        final_time=(2, 5, 4),
        time_min=(1, 3, 0.1),
        time_max=(2, 4, 0.8),
        n_shooting=(20,),
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 10826.616, decimal=3)

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        npt.assert_equal(g.shape, (120 * 5 + 7, 1))
        npt.assert_almost_equal(
            g, np.concatenate((np.zeros((120 * 5, 1)), np.array([[0, 0, 0, 0, 0, 0, 1]]).T)), decimal=6
        )
    else:
        npt.assert_equal(g.shape, (127, 1))
        npt.assert_almost_equal(g, np.concatenate((np.zeros((126, 1)), [[1]])), decimal=6)

    # Check some results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]
    tf = sol.decision_time(to_merge=SolutionMerge.NODES)[-1, 0]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    npt.assert_almost_equal(q[:, -1], np.array((2, 0, 0)))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((5.71428583, 9.81, 0)), decimal=5)
    npt.assert_almost_equal(tau[:, -1], np.array((-5.71428583, 9.81, 0)), decimal=5)

    # optimized time
    npt.assert_almost_equal(tf, 1.0, decimal=5)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_multiphase_time_constraint(ode_solver, phase_dynamics):
    # Load time_constraint
    from bioptim.examples.toy_examples.optimal_time_ocp import multiphase_time_constraint as ocp_module

    # For reducing time phase_dynamics=PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.COLLOCATION:
        return

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube.bioMod",
        final_time=(2, 5, 4),
        time_min=(1, 3, 0.1),
        time_max=(2, 4, 0.8),
        n_shooting=(20, 30, 20),
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 53441.6, decimal=1)

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        npt.assert_equal(g.shape, (421 * 5 + 22, 1))
        npt.assert_almost_equal(
            g,
            np.concatenate((np.zeros((612, 1)), [[1]], np.zeros((909, 1)), [[3]], np.zeros((603, 1)), [[1.06766639]])),
            decimal=6,
        )
    else:
        npt.assert_equal(g.shape, (447, 1))
        npt.assert_almost_equal(
            g,
            np.concatenate((np.zeros((132, 1)), [[1]], np.zeros((189, 1)), [[3]], np.zeros((123, 1)), [[1.06766639]])),
            decimal=6,
        )

    # Check some results
    states = sol.stepwise_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
    controls = sol.stepwise_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]
    tf_all = [t[-1, 0] for t in sol.decision_time(to_merge=SolutionMerge.NODES, continuous=False)]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    npt.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((5.71428583, 9.81, 0)), decimal=5)
    npt.assert_almost_equal(tau[:, -1], np.array((-5.01292039, 9.81, -7.87028502)), decimal=5)

    # optimized time
    npt.assert_almost_equal(tf_all, [1.0, 3, 1.06766639], decimal=5)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_multiphase_time_constraint_with_phase_time_equality(ode_solver, phase_dynamics):
    # Load time_constraint
    from bioptim.examples.toy_examples.optimal_time_ocp import multiphase_time_constraint as ocp_module

    # For reducing time phase_dynamics == PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.COLLOCATION:
        return

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube.bioMod",
        final_time=(2, 5, 4),
        time_min=(0.7, 3, 0.1),
        time_max=(2, 4, 1),
        n_shooting=(20, 30, 20),
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        with_phase_time_equality=True,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        npt.assert_almost_equal(f[0, 0], 53463.26239909639)
        npt.assert_equal(g.shape, (421 * 5 + 22, 1))
        npt.assert_almost_equal(
            g,
            np.concatenate(
                (np.zeros((612, 1)), [[0.95655144]], np.zeros((909, 1)), [[3]], np.zeros((603, 1)), [[0.95655144]])
            ),
            decimal=6,
        )
    else:
        npt.assert_almost_equal(f[0, 0], 53463.26240909248, decimal=1)
        npt.assert_equal(g.shape, (447, 1))
        npt.assert_almost_equal(
            g,
            np.concatenate(
                (np.zeros((132, 1)), [[0.95655144]], np.zeros((189, 1)), [[3]], np.zeros((123, 1)), [[0.95655144]])
            ),
            decimal=6,
        )

    # Check some results
    states = sol.stepwise_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
    controls = sol.stepwise_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]
    tf_all = [t[-1, 0] for t in sol.decision_time(to_merge=SolutionMerge.NODES, continuous=False)]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    npt.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((6.24518474, 9.81, 0)))
    npt.assert_almost_equal(tau[:, -1], np.array((-6.24518474, 9.81, -9.80494005)))

    # optimized time
    npt.assert_almost_equal(tf_all, [0.95655144, 3, 0.95655144], decimal=5)

    # simulate
    TestUtils.simulate(sol)


def partial_ocp_parameters(n_phases, phase_dynamics):
    if n_phases != 1 and n_phases != 3:
        raise RuntimeError("n_phases should be 1 or 3")

    biorbd_model_path = TestUtils.bioptim_folder() + "/examples/models/cube.bioMod"
    bio_model = (
        TorqueBiorbdModel(biorbd_model_path),
        TorqueBiorbdModel(biorbd_model_path),
        TorqueBiorbdModel(biorbd_model_path),
    )
    n_shooting = (2, 2, 2)
    final_time = (2, 5, 4)
    time_min = [1, 3, 0.1]
    time_max = [2, 4, 0.8]
    tau_min, tau_max, tau_init = -100, 100, 0
    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(phase_dynamics=phase_dynamics))
    if n_phases > 1:
        dynamics.add(DynamicsOptions(phase_dynamics=phase_dynamics))
        dynamics.add(DynamicsOptions(phase_dynamics=phase_dynamics))

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model[0].bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model[0].bounds_from_ranges("qdot")
    if n_phases > 1:
        x_bounds.add("q", bio_model[1].bounds_from_ranges("q"), phase=1)
        x_bounds.add("qdot", bio_model[1].bounds_from_ranges("qdot"), phase=1)
        x_bounds.add("q", bio_model[2].bounds_from_ranges("q"), phase=2)
        x_bounds.add("qdot", bio_model[2].bounds_from_ranges("qdot"), phase=2)

    for bounds in x_bounds:
        bounds["q"][1, [0, -1]] = 0
        bounds["qdot"][:, [0, -1]] = 0
    x_bounds[0]["q"][2, 0] = 0.0
    if n_phases > 1:
        x_bounds[2]["q"][2, [0, -1]] = [0.0, 1.57]

    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model[0].nb_q
    x_init["qdot"] = [0] * bio_model[0].nb_qdot
    if n_phases > 1:
        x_init.add("q", [0] * bio_model[1].nb_q, phase=1)
        x_init.add("qdot", [0] * bio_model[1].nb_qdot, phase=1)
        x_init.add("q", [0] * bio_model[2].nb_q, phase=2)
        x_init.add("qdot", [0] * bio_model[2].nb_qdot, phase=2)

    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau
    if n_phases > 1:
        u_bounds.add(
            "tau", min_bound=[tau_min] * bio_model[1].nb_tau, max_bound=[tau_max] * bio_model[1].nb_tau, phase=1
        )
        u_bounds.add(
            "tau", min_bound=[tau_min] * bio_model[2].nb_tau, max_bound=[tau_max] * bio_model[2].nb_tau, phase=2
        )

    u_init = InitialGuessList()
    u_init["tau"] = [tau_init] * bio_model[0].nb_tau
    if n_phases > 1:
        u_init.add("tau", [tau_init] * bio_model[1].nb_tau, phase=1)
        u_init.add("tau", [tau_init] * bio_model[2].nb_tau, phase=2)

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
    ) = partial_ocp_parameters(n_phases=1, phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, minimum=time_min[0], maximum=time_max[0])

    with pytest.raises(RuntimeError, match="Time constraint/objective cannot be declared more than once"):
        OptimalControlProgram(
            bio_model,
            n_shooting,
            final_time,
            dynamics=dynamics,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objective_functions,
            constraints=constraints,
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
    ) = partial_ocp_parameters(n_phases=3, phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=0)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, minimum=time_min[0], maximum=time_max[0])

    with pytest.raises(RuntimeError, match="Time constraint/objective cannot be declared more than once"):
        OptimalControlProgram(
            bio_model,
            n_shooting,
            final_time,
            dynamics=dynamics,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objective_functions,
            constraints=constraints,
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
    ) = partial_ocp_parameters(n_phases=3, phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=2)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=2)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, minimum=time_min[0], maximum=time_max[0], phase=2)

    with pytest.raises(RuntimeError, match="Time constraint/objective cannot be declared more than once"):
        OptimalControlProgram(
            bio_model,
            n_shooting,
            final_time,
            dynamics=dynamics,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objective_functions,
            constraints=constraints,
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_mayer_multiphase_time_constraint(phase_dynamics):
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
    ) = partial_ocp_parameters(n_phases=3, phase_dynamics=phase_dynamics)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=0)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=2)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, minimum=time_min[0], maximum=time_max[0], phase=2)

    OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        constraints=constraints,
    )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_mayer_neg_two_objectives(phase_dynamics):
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
    ) = partial_ocp_parameters(n_phases=1, phase_dynamics=phase_dynamics)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, phase=0)

    with pytest.raises(RuntimeError, match="Time constraint/objective cannot be declared more than once"):
        OptimalControlProgram(
            bio_model,
            n_shooting,
            final_time,
            dynamics=dynamics,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objective_functions,
        )
