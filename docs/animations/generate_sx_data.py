"""
use_sx=False (MX, default) vs use_sx=True (SX): REAL timings of the OCP build and of the IPOPT solve, same OCP.
Cart-pendulum of ``pendulum.bioMod`` (sliding translation actuated, rotation 0 -> 1 rad in T = 1 s), RK4, minimise the
integral of tau^2, IPOPT exact Hessian (default), N in {30, 50}. Each configuration is rebuilt and re-solved REPEATS times
(same process, one warm-up build+solve first, not counted). Stored in ``data/sx_timings.npz``:
    build_*  : wall time of ``OptimalControlProgram(...)``
    solve_*  : ``sol.real_time_to_optimize`` (IPOPT wall time reported by the solver interface)
    wall_*   : wall time of ``ocp.solve(solver)`` (includes the NLP function creation)
    iters_*, cost_*, status_*, nvar_*
Timings depend on the machine and are noisy; iterations and cost are the reliable columns.

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_sx_data.py
"""

import time
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
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
ROT, T, TARGET = 1, 1.0, 1.0
REPEATS = 3
CONFIGS = {"mx": dict(use_sx=False), "sx": dict(use_sx=True)}


def prepare_ocp(n, use_sx):
    bio_model = TorqueBiorbdModel(MODEL)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][ROT, -1] = TARGET
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * bio_model.nb_tau, [100] * bio_model.nb_tau
    u_bounds["tau"][ROT, :] = 0
    x_init = InitialGuessList()
    x_init.add("q", np.array([[0, 0], [0, TARGET]]), interpolation=InterpolationType.LINEAR)
    return OptimalControlProgram(
        bio_model,
        n,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4()),
        x_bounds=x_bounds,
        x_init=x_init,
        u_bounds=u_bounds,
        objective_functions=Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"),
        use_sx=use_sx,
    )


def run(n, use_sx):
    t0 = time.perf_counter()
    ocp = prepare_ocp(n, use_sx)
    build = time.perf_counter() - t0
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(500)
    t0 = time.perf_counter()
    sol = ocp.solve(solver)
    wall = time.perf_counter() - t0
    return dict(
        build=build,
        solve=float(sol.real_time_to_optimize),
        wall=wall,
        iters=int(sol.iterations),
        cost=float(sol.cost),
        status=int(sol.status),
        nvar=int(np.array(sol.vector).size),
    )


def main():
    res = {}
    for n in (30, 50):
        run(n, False), run(n, True)  # warm-up (imports, biorbd, casadi JIT-less caches)
        runs = {name: [] for name in CONFIGS}
        for _ in range(REPEATS):  # interleave the configurations to share machine load fluctuations
            for name, kw in CONFIGS.items():
                runs[name].append(run(n, **kw))
        for name, rs in runs.items():
            for key in rs[0]:
                res[f"{name}_n{n}_{key}"] = np.array([r[key] for r in rs])
            med = {k: np.median([r[k] for r in rs]) for k in ("build", "solve", "wall")}
            print(
                f"N={n} {name}: build {[round(r['build'], 2) for r in rs]} solve {[round(r['solve'], 3) for r in rs]} "
                f"wall {[round(r['wall'], 3) for r in rs]} iters {[r['iters'] for r in rs]} "
                f"cost {[round(r['cost'], 6) for r in rs]} status {[r['status'] for r in rs]} nvar {rs[0]['nvar']} "
                f"| median build {med['build']:.2f} solve {med['solve']:.3f}"
            )
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "sx_timings.npz", repeats=REPEATS, final_time=T, **res)


if __name__ == "__main__":
    main()
