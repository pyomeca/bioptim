"""
Real IPOPT data for ``anim_online.py`` (what the online plot of ``show_online_optim=True`` shows).

A live GUI window cannot be captured, so the animation re-draws the online panel from the REAL intermediate iterates:
the k-th IPOPT iterate is obtained with ``set_maximum_iterations(k)`` (IPOPT is deterministic), k = 0..n. The convergence
history (objective, inf_pr, inf_du) is read from the IPOPT stats of the full solve. One real matplotlib figure produced by
bioptim itself (``sol.graphs(show_bounds=True, save_name=...)`` on the Agg backend) is also saved.

Problem: pendulum swing-up (pendulum.bioMod, q = (y, theta), only y actuated), N = 30, T = 1 s, minimise tau, end upright
at rest. A custom plot is declared with ``ocp.add_plot`` (the angle in degrees) and ``add_plot_ipopt_outputs``.

Usage (env with bioptim), from the repo root:  PYTHONPATH=. python docs/animations/generate_online_data.py
Writes data/online_iterates.npz and data/online_graphs_*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from bioptim import (
    BoundsList,
    DynamicsOptions,
    Objective,
    ObjectiveFcn,
    OdeSolver,
    OptimalControlProgram,
    PlotType,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
ROT = 1
N, T = 30, 1.0
TAU_MAX = 100.0


def build_ocp():
    bio_model = TorqueBiorbdModel(MODEL)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"][:, 0] = 0
    x_bounds["q"][ROT, -1] = 3.14
    x_bounds["qdot"][:, -1] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-TAU_MAX] * bio_model.nb_tau, [TAU_MAX] * bio_model.nb_tau
    u_bounds["tau"][ROT, :] = 0
    ocp = OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=5)),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"),
        use_sx=True,
    )
    # custom plot: pendulum angle in degrees (same signature as bioptim/examples/getting_started/custom_plotting.py)
    ocp.add_plot(
        "angle (deg)",
        lambda t0, phases_dt, node_idx, x, u, p, a, d: x[[ROT], :] * 180 / np.pi,
        plot_type=PlotType.PLOT,
    )
    ocp.add_plot_ipopt_outputs()
    return ocp


def solve(ocp, max_iter=500):
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(max_iter)
    return ocp.solve(solver)


def traj(sol):
    st, co = sol.decision_states(), sol.decision_controls()
    q = np.array([st["q"][k][:, 0] for k in range(N + 1)]).T
    tau = np.array([co["tau"][k][:, 0] for k in range(N)]).T
    return q, tau


def save_graphs(full):
    # one real figure from bioptim (offline equivalent of the online panel), non-interactive backend
    import matplotlib.pyplot as plt

    full.graphs(automatically_organize=False, show_bounds=True, show_now=False, save_name=str(OUT / "online_graphs"))
    print("figures:", plt.get_figlabels())


if __name__ == "__main__":
    import sys

    OUT.mkdir(exist_ok=True)
    full = solve(build_ocp())
    if len(sys.argv) > 1 and sys.argv[1] == "graphs":  # only the figure (fast)
        save_graphs(full)
        sys.exit()
    stats = full.ocp.ocp_solver.shaked_ocp_solver.stats()
    it = stats["iterations"]
    n = int(full.iterations)
    print("full: iters", n, "status", full.status, "cost", full.cost, "exit", stats["return_status"])
    data = dict(
        t=np.linspace(0, T, N + 1),
        final_time=T,
        n_iter=n,
        hist_obj=np.array(it["obj"]),
        hist_inf_pr=np.array(it["inf_pr"]),
        hist_inf_du=np.array(it["inf_du"]),
        full_cost=float(full.cost),
        full_exit=str(stats["return_status"]),
    )
    q_all, tau_all, cost_all = [], [], []
    for k in range(n + 1):
        s = full if k == n else solve(build_ocp(), max_iter=k)
        q, tau = traj(s)
        q_all.append(q[ROT])
        tau_all.append(tau[0])
        cost_all.append(float(s.cost))
        print(f"k={k}: cost={cost_all[-1]:.4g} hist_obj={it['obj'][k]:.4g} inf_pr={it['inf_pr'][k]:.3g}")
    data.update(q=np.array(q_all), tau=np.array(tau_all), cost=np.array(cost_all))
    np.savez(OUT / "online_iterates.npz", **data)

    save_graphs(full)
