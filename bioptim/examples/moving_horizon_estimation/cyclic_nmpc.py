"""
In this example, nmpc (Nonlinear model predictive control) is applied on a simple 2-dofs arm model. The goal is to
perform a rotation of the arm in a quasi-cyclic manner. The sliding window across iterations is advanced for a full
cycle at a time (main difference between cyclic and normal NMPC where NMPC advance for a single frame).
"""

import platform

import numpy as np
from bioptim import (
    BiorbdModel,
    CyclicNonlinearModelPredictiveControl,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    Solver,
    Node,
    Axis,
    Solution,
    PhaseDynamics,
)


class MyCyclicNMPC(CyclicNonlinearModelPredictiveControl):
    def advance_window_bounds_states(self, sol, **extra):
        # Reimplementation of the advance_window method so the rotation of the wheel restart at -pi
        super(MyCyclicNMPC, self).advance_window_bounds_states(sol)
        self.nlp[0].x_bounds["q"][0, 0] = -np.pi
        return True


def prepare_nmpc(model_path, cycle_len, cycle_duration, max_torque, phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE, expand_dynamics=True):
    model = BiorbdModel(model_path)
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics)

    x_bound = BoundsList()
    x_bound["q"] = model.bounds_from_ranges("q")
    x_bound["qdot"] = model.bounds_from_ranges("qdot")
    u_bound = BoundsList()
    u_bound["tau"] = [-max_torque] * model.nb_q, [max_torque] * model.nb_q

    new_objectives = Objective(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q")

    # Rotate the wheel and force the marker of the hand to follow the marker on the wheel
    wheel_target = np.linspace(-np.pi, np.pi, cycle_len + 1)[np.newaxis, :]
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
        cycle_len,
        cycle_duration,
        objective_functions=new_objectives,
        constraints=constraints,
        x_bounds=x_bound,
        u_bounds=u_bound,
    )


def main():
    model_path = "models/arm2.bioMod"
    torque_max = 50

    cycle_duration = 1
    cycle_len = 20
    n_cycles = 3

    nmpc = prepare_nmpc(model_path, cycle_len=cycle_len, cycle_duration=cycle_duration, max_torque=torque_max)

    def update_functions(_nmpc: CyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
        return cycle_idx < n_cycles  # True if there are still some cycle to perform

    # Solve the program
    sol = nmpc.solve(update_functions, solver=Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
    sol.graphs()
    sol.print_cost()
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
