"""
Real IPOPT data for ``anim_solver.py`` (how IPOPT converges, and multi-start).

Problem: the pendulum swing-up of ``generate_features_data.py`` (bioptim/examples/models/pendulum.bioMod, q = (translation y,
rotation theta), only the translation is actuated, N = 30, T = 1 s, minimize the torque, end upright at rest).

    solver_iterates.npz   the trajectory after k IPOPT iterations (k = 0, 1, 2, 3, 5, 8, 12, ..., final). Each entry is a
                          separate solve with ``set_maximum_iterations(k)`` (IPOPT is deterministic, so this is the
                          k-th iterate), plus the full per-iteration history (objective, inf_pr, inf_du) read from the
                          IPOPT stats of the full solve.
    solver_multistart.npz several random initial guesses (q and tau drawn uniformly, fixed seeds), each solved to
                          convergence; final cost, status, iterations and the trajectory are stored.

Usage (env with bioptim), from the repo root:  PYTHONPATH=. python docs/animations/generate_solver_data.py
"""

import sys
from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    Objective,
    ObjectiveFcn,
    OdeSolver,
    OptimalControlProgram,
    SolutionMerge,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
ROT = 1
N, T = 30, 1.0
TAU_MAX = 100.0
ITER_LIST = [0, 1, 2, 3, 5, 8, 12, 20, 40]
N_STARTS = 12


def build_ocp(x_init_q=None, u_init=None):
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
    x_init, u_in = InitialGuessList(), InitialGuessList()
    each = InterpolationType.EACH_FRAME
    if x_init_q is not None:
        x_init.add("q", x_init_q, interpolation=each)
        u_in.add("tau", u_init, interpolation=each)
    return OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=5)),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_in,
        objective_functions=Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"),
        use_sx=True,
    )


def solve(ocp, max_iter=500, tol=None):
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(max_iter)
    if tol is not None:
        solver.set_tol(tol)
    return ocp.solve(solver)


def traj(sol):
    st, co = sol.decision_states(), sol.decision_controls()
    q = np.array([st["q"][k][:, 0] for k in range(N + 1)]).T
    tau = np.array([co["tau"][k][:, 0] for k in range(N)]).T
    return q, tau


def stats(sol):
    return sol.ocp.ocp_solver.shaked_ocp_solver.stats()


def random_start(rng):
    """Smooth random guess: linear ramp to the target plus random low-frequency sines on both q, small random tau."""
    tt = np.linspace(0, 1, N + 1)
    q = np.zeros((2, N + 1))
    q[ROT] = 3.14 * tt
    for dof in range(2):
        for f in (1, 2, 3):
            q[dof] += rng.uniform(-1.2, 1.2) * np.sin(np.pi * f * tt)
    q[:, 0] = 0
    tau = rng.uniform(-30, 30, size=(2, N))
    tau[ROT] = 0
    return q, tau


if __name__ == "__main__":
    explore = len(sys.argv) > 1 and sys.argv[1] == "explore"
    OUT.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- part 1: iterates
    print("Part 1: iterates")
    full = solve(build_ocp())
    st = stats(full)
    it = st["iterations"]
    n_full = int(full.iterations)
    print("  full: iters", n_full, "status", full.status, "cost", full.cost, "exit", st["return_status"])
    data = dict(
        n_shooting=N,
        final_time=T,
        hist_obj=np.array(it["obj"]),
        hist_inf_pr=np.array(it["inf_pr"]),
        hist_inf_du=np.array(it["inf_du"]),
        full_iterations=n_full,
        full_cost=float(full.cost),
        full_exit=str(st["return_status"]),
    )
    ks = [k for k in ITER_LIST if k < n_full] + [n_full]
    data["ks"] = np.array(ks)
    for i, k in enumerate(ks):
        s = full if k == n_full else solve(build_ocp(), max_iter=k)
        q, tau = traj(s)
        data[f"q_{i}"], data[f"tau_{i}"] = q, tau
        data[f"cost_{i}"] = float(s.cost)
        # objective / infeasibility of the k-th iterate come from the history of the full solve
        print(
            f"  k={k}: cost={float(s.cost):.4g}  hist obj={it['obj'][min(k, n_full)]:.4g} inf_pr={it['inf_pr'][min(k, n_full)]:.3g}"
        )
    data["t"] = np.linspace(0, T, N + 1)
    np.savez(OUT / "solver_iterates.npz", **data)

    # ---------------------------------------------------------------- part 2: multi-start
    print("Part 2: multi-start")
    rng = np.random.default_rng(0)
    res = []
    for s in range(N_STARTS):
        q0, u0 = random_start(rng)
        try:
            sol = solve(build_ocp(q0, u0), max_iter=500)
            q, tau = traj(sol)
            res.append((s, q0, q, tau, float(sol.cost), int(sol.status), int(sol.iterations)))
            print(
                f"  start {s}: status={sol.status} iters={sol.iterations} cost={float(sol.cost):.5g}  q_rot max={q[ROT].max():.2f} min={q[ROT].min():.2f}"
            )
        except Exception as e:
            print(f"  start {s}: failed {e}")
    d2 = dict(n_starts=len(res), t=np.linspace(0, T, N + 1))
    for i, (s, q0, q, tau, c, stt, its) in enumerate(res):
        d2.update({f"q0_{i}": q0, f"q_{i}": q, f"tau_{i}": tau, f"cost_{i}": c, f"status_{i}": stt, f"iters_{i}": its})
    np.savez(OUT / "solver_multistart.npz", **d2)
