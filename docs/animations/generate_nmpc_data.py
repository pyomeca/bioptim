"""
Real NMPC (bioptim.NonlinearModelPredictiveControl) on the cart-pendulum of the MHE examples
(bioptim/examples/toy_examples/moving_horizon_estimation/mhe.py uses the same model). The cart must follow a cyclic
reference (sine, period 2 s) with a 1 s horizon (10 nodes of 0.1 s); the window is advanced by ONE node per solve.
Every window is a real IPOPT solve; everything is stored in data/nmpc_results.npz.

Run from the repo root:  PYTHONPATH=. python docs/animations/generate_nmpc_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    DynamicsOptions,
    Node,
    NonlinearModelPredictiveControl,
    Objective,
    ObjectiveFcn,
    ObjectiveList,
    Solver,
    SolutionMerge,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

OUT = Path(__file__).parent / "data"
MODEL = ExampleUtils.folder + "/models/cart_pendulum.bioMod"

N = 10  # window_len (shooting intervals in the window)
DT = 0.1
WINDOW_DURATION = N * DT
N_STEPS = 30  # number of NMPC iterations = 3 s of closed-loop motion
AMP, PERIOD = 0.5, 2.0
TAU_MAX = 60.0


def reference(t):
    return AMP * np.sin(2 * np.pi * t / PERIOD)


def main():
    model = TorqueBiorbdModel(MODEL)
    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["q"][:, 0] = 0  # initial position
    x_bounds["qdot"][:, 0] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-TAU_MAX, 0.0], [TAU_MAX, 0.0]  # only the cart is actuated

    objectives = ObjectiveList()
    objectives.add(  # list_index 0: the moving target
        ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=0, node=Node.ALL, weight=1000, target=np.zeros((1, N + 1))
    )
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.01)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1)

    nmpc = NonlinearModelPredictiveControl(
        model,
        window_len=N,
        window_duration=WINDOW_DURATION,
        dynamics=DynamicsOptions(),
        common_objective_functions=objectives,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
    )

    def update_function(_nmpc, step, _sol):
        t0 = step * DT
        _nmpc.update_objectives_target(target=reference(t0 + DT * np.arange(N + 1))[None, :], list_index=0)
        return step < N_STEPS

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    solver.set_maximum_iterations(200)
    sol, all_sols, _ = nmpc.solve(update_function, solver=solver, get_all_iterations=True)

    out = dict(n=N, dt=DT, n_steps=N_STEPS, amp=AMP, period=PERIOD, tau_max=TAU_MAX)
    pred_q, pred_th, pred_tau, status, iters = [], [], [], [], []
    for s in all_sols:
        st = s.decision_states(to_merge=SolutionMerge.NODES)
        ct = s.decision_controls(to_merge=SolutionMerge.NODES)
        pred_q.append(st["q"][0, :])
        pred_th.append(st["q"][1, :])
        pred_tau.append(ct["tau"][0, :N])
        status.append(s.status)
        iters.append(s.iterations)
    out["pred_q"] = np.array(pred_q)  # (steps, N+1) cart position, predicted over each window
    out["pred_theta"] = np.array(pred_th)
    out["pred_tau"] = np.array(pred_tau)  # (steps, N)
    out["status"] = np.array(status)
    out["iterations"] = np.array(iters)
    fs = sol.decision_states(to_merge=SolutionMerge.NODES)
    fc = sol.decision_controls(to_merge=SolutionMerge.NODES)
    out["applied_q"] = fs["q"][0, :]  # (steps+1)
    out["applied_theta"] = fs["q"][1, :]
    out["applied_tau"] = fc["tau"][0, :]  # (steps)
    np.savez(OUT / "nmpc_results.npz", **out)
    print("status", out["status"], "iterations", out["iterations"])
    print("applied_q", out["applied_q"].shape, "applied_tau", out["applied_tau"].shape)
    err = out["applied_q"][:-1] - reference(np.arange(N_STEPS) * DT)
    print("max tracking error (m)", np.abs(err).max())


if __name__ == "__main__":
    main()
