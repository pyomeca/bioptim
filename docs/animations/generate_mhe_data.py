"""
Real Moving Horizon Estimation (bioptim.MovingHorizonEstimator) on the cart-pendulum of the MHE example
(bioptim/examples/toy_examples/moving_horizon_estimation/mhe.py). Ground truth: a simulated motion (solve_ivp) driven by
a known cart force.  Measurement: the pendulum angle (theta) + Gaussian noise (SYNTHETIC, seed 0).  The estimator
window has N = 10 intervals of 0.05 s (0.5 s); it slides by one measurement per solve.  Every window is a real IPOPT
solve; everything is stored in data/mhe_results.npz.

Run from the repo root:  PYTHONPATH=. python docs/animations/generate_mhe_data.py
"""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from bioptim import (
    BoundsList,
    DynamicsOptions,
    MovingHorizonEstimator,
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

N = 10  # window_len
DT = 0.05
WINDOW_DURATION = N * DT
N_STEPS = 40  # number of windows (measurements 0 .. N_STEPS + N)
NOISE_STD = 0.1  # rad
TAU_MAX = 5.0
SEED = 0


def main():
    model = TorqueBiorbdModel(MODEL)
    n_frames = N_STEPS + N + 1
    t = DT * np.arange(n_frames)

    # --------------------------------------------------------------------------------------------- synthetic truth
    qddot = model.forward_dynamics()

    def ode(_t, x, u):
        return np.concatenate((x[2:, None], np.array(qddot(x[:2], x[2:], u, [], [])))).ravel()

    tau_true = 2.0 * np.sin(2 * np.pi * t / 1.5)  # known cart force
    x = np.array([0.0, np.pi / 2, 0.0, 0.0])
    truth = np.zeros((4, n_frames))
    for k in range(n_frames):
        truth[:, k] = x
        u = np.array([tau_true[k], 0.0])
        x = solve_ivp(ode, [0, DT], x, args=(u,), rtol=1e-9, atol=1e-11).y[:, -1]
    rng = np.random.default_rng(SEED)
    meas = truth[1] + NOISE_STD * rng.standard_normal(n_frames)

    # --------------------------------------------------------------------------------------------------- estimator
    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    u_bounds = BoundsList()
    u_bounds["tau"] = [-TAU_MAX, 0.0], [TAU_MAX, 0.0]

    objectives = ObjectiveList()
    objectives.add(  # list_index 0: the moving measurement
        ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=1, node=Node.ALL, weight=1000, target=np.zeros((1, N + 1))
    )
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1.0)

    mhe = MovingHorizonEstimator(
        model,
        window_len=N,
        window_duration=WINDOW_DURATION,
        dynamics=DynamicsOptions(),
        common_objective_functions=objectives,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
    )

    def update_function(_mhe, step, _sol):
        _mhe.update_objectives_target(target=meas[None, step : step + N + 1], list_index=0)
        return step < N_STEPS

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    solver.set_maximum_iterations(200)
    sol, all_sols, _ = mhe.solve(update_function, solver=solver, get_all_iterations=True)

    pred = []
    status, iters = [], []
    for s in all_sols:
        st = s.decision_states(to_merge=SolutionMerge.NODES)
        pred.append(st["q"][1, :])
        status.append(s.status)
        iters.append(s.iterations)
    fs = sol.decision_states(to_merge=SolutionMerge.NODES)
    # returned estimate: first node of each window. The trailing element appended by solve() (last node of the
    # last window, but read from a different container) is dropped so that est[k] is the estimate at t = k * DT.
    est = fs["q"][1, :N_STEPS]

    out = dict(
        n=N,
        dt=DT,
        n_steps=N_STEPS,
        noise_std=NOISE_STD,
        t=t,
        truth_theta=truth[1],
        meas_theta=meas,
        pred_theta=np.array(pred),
        est_theta=est,
        status=np.array(status),
        iterations=np.array(iters),
    )
    np.savez(OUT / "mhe_results.npz", **out)
    print("status", out["status"], "\niterations", out["iterations"])
    print("est shape", est.shape)
    m = len(est)
    print("returned estimate length", len(fs["q"][1, :]), "used", m)
    e_est = est - truth[1, :m]
    e_meas = meas[:m] - truth[1, :m]
    print("RMS est err", np.sqrt(np.mean(e_est**2)), "RMS meas err", np.sqrt(np.mean(e_meas**2)))
    last = np.array([p[-1] for p in pred])
    print("RMS err of last-node (filter) estimate", np.sqrt(np.mean((last - truth[1, N : N + N_STEPS]) ** 2)))


if __name__ == "__main__":
    main()
