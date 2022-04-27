"""
In this example, nmpc (Nonlinear model predictive control) is applied on a simple 2-dofs arm model. The goal is to
perform a rotation of the arm in a quasi-cyclic manner. The sliding window across iterations is advanced for a full
cycle at a time (main difference between cyclic and normal NMPC where NMPC advance for a single frame).
"""

import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    CyclicNonlinearModelPredictiveControl,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    Solver,
    Node,
    Axis,
    Solution,
)


class MyCyclicNMPC(CyclicNonlinearModelPredictiveControl):
    def advance_window_bounds_states(self, sol):
        # Reimplementation of the advance_window method so the rotation of the wheel restart at -pi
        super(MyCyclicNMPC, self).advance_window_bounds_states(sol)
        self.nlp[0].x_bounds[0, 0] = -np.pi
        return True


def prepare_nmpc(model_path, cycle_len, cycle_duration, max_torque):
    model = biorbd.Model(model_path)
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    x_bound = QAndQDotBounds(model)
    u_bound = Bounds([-max_torque] * model.nbQ(), [max_torque] * model.nbQ())

    x_init = InitialGuess(
        np.zeros(
            model.nbQ() * 2,
        )
    )
    u_init = InitialGuess(
        np.zeros(
            model.nbQ(),
        )
    )

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
        x_init=x_init,
        u_init=u_init,
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
    sol = nmpc.solve(update_functions, solver=Solver.IPOPT(show_online_optim=True))
    sol.graphs()
    sol.print()
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
