"""
In this example, nmpc (Nonlinear model predictive control) is applied on a simple 2-dofs arm model. The goal is to
perform a rotation of the arm in a quasi-cyclic manner. The sliding window across iterations is advanced for a full
cycle at a time while optimizing three cycles at a time (main difference between cyclic and multi-cyclic is that
the latter has more cycle at a time giving the knowledge to the solver that 'something' is coming after)
"""

import platform

import numpy as np
from bioptim import (
    BiorbdModel,
    MultiCyclicNonlinearModelPredictiveControl,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    PhaseDynamics,
    Solver,
    Node,
    Axis,
    Solution,
    SolutionMerge,
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


def prepare_nmpc(
    model_path,
    cycle_len,
    cycle_duration,
    n_cycles_simultaneous,
    n_cycles_to_advance,
    max_torque,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
):
    model = BiorbdModel(model_path)
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics)

    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"].min[0, :] = -2 * np.pi * n_cycles_simultaneous  # Allow the wheel to spin as much as needed
    x_bounds["q"].max[0, :] = 0
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")

    u_bounds = BoundsList()
    u_bounds["tau"] = [-max_torque] * model.nb_q, [max_torque] * model.nb_q

    new_objectives = Objective(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q")

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
        objective_functions=new_objectives,
        constraints=constraints,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
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
    )
    sol.print_cost()
    sol.graphs()
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
