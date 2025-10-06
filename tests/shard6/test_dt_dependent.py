"""
This example uses the data from the balanced pendulum example to generate the data to track.
When it optimizes the program, contrary to the vanilla pendulum, it tracks the values instead of 'knowing' that
it is supposed to balance the pendulum. It is designed to show how to track marker and kinematic data.

Note that the final node is not tracked.
"""

from bioptim import (
    BiorbdModel,
    BoundsList,
    TorqueDynamics,
    ControlType,
    DynamicsEvaluation,
    DynamicsFunctions,
    DynamicsOptionsList,
    InitialGuessList,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    NonLinearProgram,
    PhaseDynamics,
    SolutionMerge,
)
from casadi import MX, SX, vertcat, sin, Function, DM
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils


class CustomModel(BiorbdModel, TorqueDynamics):
    def __init__(self, bio_model: str):
        BiorbdModel.__init__(self, bio_model)
        self.fatigue = None
        TorqueDynamics.__init__(self)

    def dynamics(
        self,
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        algebraic_states: MX | SX,
        numerical_timeseries: MX | SX,
        nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls) * (
            sin(nlp.tf - time) * time.ones(nlp.model.nb_tau) * 10
        )

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = nlp.model.forward_dynamics()(q, qdot, tau, [], parameters)

        return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)


def prepare_ocp_state_as_time(
    biorbd_model_path: str,
    n_phase: int,
    control_type: ControlType,
    minimize_time: bool,
    use_sx: bool,
    phase_dynamics: PhaseDynamics = PhaseDynamics.ONE_PER_NODE,
) -> OptimalControlProgram:

    bio_model = (
        [CustomModel(biorbd_model_path)]
        if n_phase == 1
        else [CustomModel(biorbd_model_path), CustomModel(biorbd_model_path)]
    )
    final_time = 1
    n_shooting = 30

    # Add objective functions
    objective_functions = ObjectiveList()
    for i in range(len(bio_model)):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=i, quadratic=True)
        if minimize_time:
            target = 1 if i == 0 else 2
            objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_TIME, target=target, weight=20000, phase=i, quadratic=True
            )

    # Dynamics
    dynamics = DynamicsOptionsList()
    for i in range(len(bio_model)):
        dynamics.add(
            phase=i,
            ode_solver=OdeSolver.RK4(n_integration_steps=5),
            expand_dynamics=True,
            phase_dynamics=phase_dynamics,
        )

    # Define states path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model[0].bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][-1, -1] = 3.14
    x_bounds["qdot"] = bio_model[0].bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    # Define control path constraint
    n_tau = bio_model[0].nb_tau
    u_bounds = BoundsList()
    u_bounds_tau = [[-100] * n_tau, [100] * n_tau]  # Limit the strength of the pendulum to (-100 to 100)...
    u_bounds_tau[0][1] = 0  # ...but remove the capability to actively rotate
    u_bounds_tau[1][1] = 0  # ...but remove the capability to actively rotate
    for i in range(len(bio_model)):
        u_bounds.add("tau", min_bound=u_bounds_tau[0], max_bound=u_bounds_tau[1], phase=i)

    x_init = InitialGuessList()
    u_init = InitialGuessList()

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        control_type=control_type,
        use_sx=use_sx,
    )


def integrate_RK4(time_vector, dt, states, controls, dyn_fun, n_shooting=30, n_steps=5):
    n_q = 2
    tf = time_vector[-1]
    h = dt / n_steps
    x_integrated = DM.zeros((n_q * 2, n_shooting + 1))
    x_integrated[:, 0] = states[:, 0]
    for i_shooting in range(n_shooting):
        x_this_time = x_integrated[:, i_shooting]
        u_this_time = controls[:, i_shooting]
        current_time = dt * i_shooting
        for i_step in range(n_steps):
            k1 = dyn_fun(current_time, tf, x_this_time, u_this_time)
            k2 = dyn_fun(current_time + h / 2, tf, x_this_time + h / 2 * k1, u_this_time)
            k3 = dyn_fun(current_time + h / 2, tf, x_this_time + h / 2 * k2, u_this_time)
            k4 = dyn_fun(current_time + h, tf, x_this_time + h * k3, u_this_time)
            x_this_time += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            current_time += h
        x_integrated[:, i_shooting + 1] = x_this_time
    return x_integrated


@pytest.mark.parametrize("minimize_time", [True, False])
@pytest.mark.parametrize("use_sx", [False, True])
def test_dt_dependent_problem(minimize_time, use_sx):

    from bioptim.examples.toy_examples.torque_driven_ocp import example_multi_biorbd_model as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Solve the program --- #
    ocp = prepare_ocp_state_as_time(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
        n_phase=1,
        control_type=ControlType.CONSTANT,
        minimize_time=minimize_time,
        use_sx=use_sx,
    )
    sol = ocp.solve()

    if minimize_time:
        npt.assert_almost_equal(np.array(sol.cost), np.array([[355.86984543]]))
        npt.assert_almost_equal(
            sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
            -0.2860762487737413,
        )
        npt.assert_almost_equal(
            sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
            0.3506686467731191,
        )
        npt.assert_almost_equal(
            sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
            1.789504566571892,
        )
        npt.assert_almost_equal(sol.decision_time()[-1], 1.05035, decimal=5)
    else:
        npt.assert_almost_equal(np.array(sol.cost), np.array([[434.53255971]]))
        npt.assert_almost_equal(
            sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
            -0.3036933491652515,
        )
        npt.assert_almost_equal(
            sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
            0.7605699746673404,
        )
        npt.assert_almost_equal(
            sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
            2.2783113520562894,
        )
        npt.assert_almost_equal(sol.decision_time()[-1], 1)

    # Test the integration
    time_sym = MX.sym("T", 1, 1)
    tf_sym = MX.sym("Tf", 1, 1)
    states_sym = MX.sym("Q_Qdot", 4, 1)
    controls_sym = MX.sym("Tau", 2, 1)

    tau_dyn = controls_sym * (sin(tf_sym - time_sym) * MX.ones(ocp.nlp[0].model.nb_tau) * 10)
    out_dyn = vertcat(
        states_sym[ocp.nlp[0].model.nb_q :],
        ocp.nlp[0].model.forward_dynamics()(
            states_sym[: ocp.nlp[0].model.nb_q], states_sym[ocp.nlp[0].model.nb_q :], tau_dyn, [], []
        ),
    )

    dyn_fun = Function(
        "dynamics",
        [time_sym, tf_sym, states_sym, controls_sym],
        [out_dyn],
    )
    sol_time_vector = sol.decision_time()
    sol_states = vertcat(
        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"],
        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["qdot"],
    )
    sol_controls = sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"]
    x_integrated = integrate_RK4(
        sol_time_vector, sol_time_vector[1], sol_states, sol_controls, dyn_fun, n_shooting=30, n_steps=5
    )

    npt.assert_almost_equal(np.array(x_integrated[:4, :]).reshape(4 * 31, 1), np.array(sol_states).reshape(4 * 31, 1))

    # Test de dynamics
    node_idx = 1
    step_idx = 1
    t_span = DM.zeros(2)
    t_span[0] = sol_time_vector[node_idx] + step_idx * sol_time_vector[1]
    t_span[1] = sol_time_vector[1]
    x = sol_states[:, node_idx]
    u = sol_controls[:, node_idx]
    dynamics_bioptim = ocp.nlp[0].dynamics_func(t_span, x, u, [], [], [])
    dynamics_hand_written = dyn_fun(t_span[0], sol_time_vector[-1], x, u)
    npt.assert_almost_equal(np.array(dynamics_bioptim), np.array(dynamics_hand_written))
