"""
In this example, nmpc (Nonlinear model predictive control) is applied on a simple 2-dofs arm model. The goal is to
perform a rotation of the arm in a quasi-cyclic manner. The sliding window across iterations is advanced for a full
cycle at a time while optimizing three cycles at a time (main difference between cyclic and multi-cyclic is that
the latter has more cycle at a time giving the knowledge to the solver that 'something' is coming after)
"""

import platform

import biorbd
import numpy as np
from casadi import MX, SX, vertcat

from bioptim import (
    Axis,
    BiorbdModel,
    BoundsList,
    ConfigureProblem,
    ConstraintFcn,
    ConstraintList,
    Dynamics,
    DynamicsEvaluation,
    DynamicsFunctions,
    InitialGuessList,
    InterpolationType,
    MultiCyclicCycleSolutions,
    MultiCyclicNonlinearModelPredictiveControl,
    Node,
    NonLinearProgram,
    ObjectiveList,
    ObjectiveFcn,
    OptimalControlProgram,
    ParameterList,
    PenaltyController,
    PhaseDynamics,
    Solution,
    SolutionMerge,
    Solver,
    VariableScaling,
    ContactType,
)


class MyCyclicNMPC(MultiCyclicNonlinearModelPredictiveControl):
    def advance_window_bounds_states(self, sol, n_cycles_simultaneous=None):
        # Reimplementation of the advance_window method so the rotation of the wheel restart at -pi
        super(MyCyclicNMPC, self).advance_window_bounds_states(sol)
        self.nlp[0].x_bounds["q"].min[0, :] = -2 * np.pi * n_cycles_simultaneous
        self.nlp[0].x_bounds["q"].max[0, :] = 0
        return True

    def advance_window_initial_guess_states(self, sol, n_cycles_simultaneous=None):
        # Reimplementation of the advance_window method so the rotation of the wheel restart at -pi
        super(MyCyclicNMPC, self).advance_window_initial_guess_states(sol)
        q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
        self.nlp[0].x_init["q"].init[0, :] = q[0, :]  # Keep the previously found value for the wheel
        return True


def dummy_parameter_function(bio_model: biorbd.Model, value: MX):
    return


def param_custom_objective(controller: PenaltyController) -> MX:
    return controller.parameters["tau_modifier"].cx - 2


def parameter_dependent_dynamic(
    time: MX | SX,
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    algebraic_states: MX | SX,
    numerical_timeseries: MX | SX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(t, x, u, p, a, d)

    Parameters
    ----------
    time: MX | SX
        The time of the system
    states: MX | SX
        The state of the system
    controls: MX | SX
        The controls of the system
    parameters: MX | SX
        The parameters acting on the system
    algebraic_states: MX | SX
        The algebraic states variables of the system
    numerical_timeseries: MX | SX
        The numerical timeseries of the system
    nlp: NonLinearProgram
        A reference to the phase

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls) * (parameters)

    # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    ddq = nlp.model.forward_dynamics(with_contact=False)(q, qdot, tau, [], [])

    return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)


def custom_configure(
    ocp: OptimalControlProgram,
    nlp: NonLinearProgram,
    numerical_data_timeseries: dict[str, np.ndarray] = None,
    contact_type: list[ContactType] = None,
):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
    """

    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_dynamics_function(ocp, nlp, parameter_dependent_dynamic)


def prepare_nmpc(
    model_path,
    cycle_len,
    cycle_duration,
    n_cycles_simultaneous,
    n_cycles_to_advance,
    max_torque,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    use_sx: bool = False,
):
    model = BiorbdModel(model_path)

    parameter = ParameterList(use_sx=use_sx)
    parameter_bounds = BoundsList()
    parameter_init = InitialGuessList()

    parameter.add(
        name="tau_modifier",
        function=dummy_parameter_function,
        size=1,
        scaling=VariableScaling("tau_modifier", [1]),
    )

    parameter_init["tau_modifier"] = np.array([2.25])

    parameter_bounds.add(
        "tau_modifier",
        min_bound=[1.5],
        max_bound=[3.5],
        interpolation=InterpolationType.CONSTANT,
    )

    dynamics = Dynamics(
        custom_configure, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics, contact_type=[]
    )

    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"].min[0, :] = -2 * np.pi * n_cycles_simultaneous  # Allow the wheel to spin as much as needed
    x_bounds["q"].max[0, :] = 0
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")

    u_bounds = BoundsList()
    u_bounds["tau"] = [-max_torque] * model.nb_q, [max_torque] * model.nb_q

    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=0, target=0, weight=100)
    objectives.add(param_custom_objective, custom_type=ObjectiveFcn.Mayer, weight=1, quadratic=True)

    # Rotate the wheel and force the marker of the hand to follow the marker on the wheel
    wheel_target = np.linspace(-2 * np.pi * n_cycles_simultaneous, 0, cycle_len * n_cycles_simultaneous + 1)[
        np.newaxis, :
    ]
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", index=0, node=Node.ALL, target=wheel_target)
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.ALL,
        first_marker="wheel",
        second_marker="COM_hand",
        axes=[Axis.X, Axis.Y],
    )

    return MyCyclicNMPC(
        model,
        dynamics,
        cycle_len=cycle_len,
        cycle_duration=cycle_duration,
        n_cycles_simultaneous=n_cycles_simultaneous,
        n_cycles_to_advance=n_cycles_to_advance,
        common_objective_functions=objectives,
        constraints=constraints,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        parameters=parameter,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        use_sx=use_sx,
    )


def main():
    model_path = "models/arm2.bioMod"
    torque_max = 50

    cycle_duration = 1
    cycle_len = 20
    n_cycles_to_advance = 1
    n_cycles_simultaneous = 3
    n_cycles = 4

    nmpc = prepare_nmpc(
        model_path,
        cycle_len=cycle_len,
        cycle_duration=cycle_duration,
        n_cycles_to_advance=n_cycles_to_advance,
        n_cycles_simultaneous=n_cycles_simultaneous,
        max_torque=torque_max,
    )

    def update_functions(_nmpc: MultiCyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
        return cycle_idx < n_cycles  # True if there are still some cycle to perform

    # Solve the program
    sol = nmpc.solve(
        update_functions,
        solver=Solver.IPOPT(show_online_optim=platform.system() == "Linux"),
        n_cycles_simultaneous=n_cycles_simultaneous,
        get_all_iterations=True,
        cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
    )
    sol[0].print_cost()
    sol[0].graphs()
    sol[0].animate(n_frames=100)


if __name__ == "__main__":
    main()
