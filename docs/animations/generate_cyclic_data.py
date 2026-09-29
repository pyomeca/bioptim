"""
Real cyclic NMPC (bioptim.CyclicNonlinearModelPredictiveControl) on the cart-pendulum of the MHE examples.
Each solve optimises ONE cycle (window = cycle, 20 nodes of 0.2 s = 4 s).  After each solve the window is advanced by a
whole cycle: the initial state of the next window is the LAST state of the previous one, and the last node is bounded
to this same state (+/- 1 % of the range): the cyclic constraint (see CyclicRecedingHorizonOptimization in
bioptim/optimization/receding_horizon_optimization.py).  The very first window has no anchor (its final state is only
bounded by the range).  The cart must follow a sine of period 4 s; its amplitude changes for the 3rd cycle.
Everything is stored in data/cyclic_results.npz.

Run from the repo root:  PYTHONPATH=. python docs/animations/generate_cyclic_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    CyclicNonlinearModelPredictiveControl,
    DynamicsOptions,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    SolutionMerge,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

OUT = Path(__file__).parent / "data"
MODEL = ExampleUtils.folder + "/models/cart_pendulum.bioMod"

CYCLE_LEN = 20
CYCLE_DURATION = 4.0
N_CYCLES = 4
AMPS = [0.4, 0.4, 0.7, 0.7]  # reference amplitude of each cycle (m)
TAU_MAX = 60.0


def reference(amp):
    t = np.linspace(0, CYCLE_DURATION, CYCLE_LEN + 1)
    return (amp * np.sin(2 * np.pi * t / CYCLE_DURATION))[None, :]


def main():
    model = TorqueBiorbdModel(MODEL)
    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    # tighter than the model ranges: the cyclic slack is 1 % of (max - min), so a narrower range = a tighter cycle
    for key, (lo, hi) in {"q": ([-1.0, -np.pi], [1.0, np.pi]), "qdot": ([-6.0, -6.0], [6.0, 6.0])}.items():
        x_bounds[key].min[:, :] = np.array(lo)[:, None]
        x_bounds[key].max[:, :] = np.array(hi)[:, None]
    u_bounds = BoundsList()
    u_bounds["tau"] = [-TAU_MAX, 0.0], [TAU_MAX, 0.0]  # only the cart is actuated

    objectives = ObjectiveList()
    objectives.add(  # list_index 0: the moving target
        ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=0, node=Node.ALL, weight=1000, target=reference(AMPS[0])
    )
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.01)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1)

    nmpc = CyclicNonlinearModelPredictiveControl(
        model,
        cycle_len=CYCLE_LEN,
        cycle_duration=CYCLE_DURATION,
        dynamics=DynamicsOptions(),
        common_objective_functions=objectives,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
    )

    def update_function(_nmpc, cycle, _sol):
        if cycle < N_CYCLES:
            _nmpc.update_objectives_target(target=reference(AMPS[cycle]), list_index=0)
        return cycle < N_CYCLES

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    solver.set_maximum_iterations(300)
    sol, all_sols, _ = nmpc.solve(update_function, solver=solver, get_all_iterations=True)

    out = dict(cycle_len=CYCLE_LEN, cycle_duration=CYCLE_DURATION, n_cycles=N_CYCLES, amps=np.array(AMPS))
    q, th, qd, tau, status, iters, cost = [], [], [], [], [], [], []
    for s in all_sols:
        st = s.decision_states(to_merge=SolutionMerge.NODES)
        ct = s.decision_controls(to_merge=SolutionMerge.NODES)
        q.append(st["q"][0])
        th.append(st["q"][1])
        qd.append(st["qdot"][0])
        tau.append(ct["tau"][0, :CYCLE_LEN])
        status.append(s.status)
        iters.append(s.iterations)
        cost.append(float(s.cost))
    out.update(
        win_q=np.array(q),
        win_theta=np.array(th),
        win_qdot=np.array(qd),
        win_tau=np.array(tau),
        status=np.array(status),
        iterations=np.array(iters),
        cost=np.array(cost),
    )
    # all four states at first / last node of every window (cyclic gap)
    out["win_x_first"] = np.array(
        [
            np.concatenate([s.decision_states(to_merge=SolutionMerge.NODES)[k][:, 0] for k in ("q", "qdot")])
            for s in all_sols
        ]
    )
    out["win_x_last"] = np.array(
        [
            np.concatenate([s.decision_states(to_merge=SolutionMerge.NODES)[k][:, -1] for k in ("q", "qdot")])
            for s in all_sols
        ]
    )
    fs = sol.decision_states(to_merge=SolutionMerge.NODES)
    out["concat_q"] = fs["q"][0]
    out["concat_qdot"] = fs["qdot"][0]
    out["concat_theta"] = fs["q"][1]
    np.savez(OUT / "cyclic_results.npz", **out)
    print("status", out["status"], "iterations", out["iterations"])
    print("concat", out["concat_q"].shape)
    gap = np.abs(out["win_x_last"] - out["win_x_first"])
    print("max |x_last - x_first| per cycle", gap.max(axis=1))


if __name__ == "__main__":
    main()
