"""
The Solution object and its accessors. ONE real solve (2-phase pendulum, multiple shooting with RK4, IPOPT), then the
same solution is read through every accessor and the real arrays are stored in ``data/solution_pendulum.npz``.

Problem: pendulum.bioMod (nq = 2, only the sliding translation is actuated), phase 0: 4 intervals over 0.6 s (RK4, 3 steps),
phase 1: 3 intervals over 0.4 s (RK4, 2 steps); q(0) = 0, q_rot(end of phase 1) = 1 rad, minimise the squared torque.

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_solution_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    DynamicsOptions,
    DynamicsOptionsList,
    InitialGuessList,
    InterpolationType,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    SolutionMerge,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
NS, TS, STEPS = (4, 3), (0.6, 0.4), (3, 2)
ROT, TARGET = 1, 1.0


def prepare_ocp():
    models = [TorqueBiorbdModel(MODEL) for _ in NS]
    x_bounds, u_bounds, x_init, dyn, obj = (
        BoundsList(),
        BoundsList(),
        InitialGuessList(),
        DynamicsOptionsList(),
        ObjectiveList(),
    )
    for p, m in enumerate(models):
        x_bounds.add("q", m.bounds_from_ranges("q"), phase=p)
        x_bounds.add("qdot", m.bounds_from_ranges("qdot"), phase=p)
        u_bounds.add("tau", min_bound=[-100, 0], max_bound=[100, 0], phase=p)  # rotation is passive
        dyn.add(DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=STEPS[p])))
        obj.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=p)
    x_bounds[0]["q"][:, 0] = 0
    x_bounds[0]["qdot"][:, 0] = 0
    x_bounds[1]["q"][ROT, -1] = TARGET
    x_bounds[1]["qdot"][:, -1] = 0
    x_init.add("q", np.array([[0, 0], [0, TARGET / 2]]), interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("q", np.array([[0, 0], [TARGET / 2, TARGET]]), interpolation=InterpolationType.LINEAR, phase=1)
    return OptimalControlProgram(
        models,
        list(NS),
        list(TS),
        dynamics=dyn,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=obj,
        use_sx=True,
    )


def main():
    ocp = prepare_ocp()
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(500)
    sol = ocp.solve(solver)
    print("status", sol.status, "iterations", sol.iterations, "cost", float(sol.cost))
    res = dict(
        status=int(sol.status),
        iterations=int(sol.iterations),
        cost=float(sol.cost),
        ns=np.array(NS),
        ts=np.array(TS),
        steps=np.array(STEPS),
    )

    def rec(name, arr):
        arr = np.asarray(arr, dtype=float)
        res[name] = arr
        print(f"{name:32s} {arr.shape}")

    # per phase, not merged: dict key -> list over nodes of (n, n_sub)
    ds = sol.decision_states()
    ss = sol.stepwise_states()
    dc = sol.decision_controls()
    sc = sol.stepwise_controls()
    print("decision_states type", type(ds), len(ds), type(ds[0]), list(ds[0].keys()))
    print("q phase0 node shapes", [a.shape for a in ds[0]["q"]])
    print("stepwise q phase0 node shapes", [a.shape for a in ss[0]["q"]])
    print("controls tau phase0 node shapes", [a.shape for a in dc[0]["tau"]], [a.shape for a in sc[0]["tau"]])
    dt = sol.decision_time()
    st = sol.stepwise_time()
    print("decision_time p0 shapes", [np.asarray(a).shape for a in dt[0]])
    print("stepwise_time p0 shapes", [np.asarray(a).shape for a in st[0]])
    # merged over nodes (per phase)
    m = [SolutionMerge.NODES]
    for name, fn in (("decision_states", sol.decision_states), ("stepwise_states", sol.stepwise_states)):
        d = fn(to_merge=m)
        for p in range(2):
            rec(f"{name}_nodes_p{p}_q", d[p]["q"])
    rec("decision_states_KEYSNODES_p0", sol.decision_states(to_merge=[SolutionMerge.KEYS, SolutionMerge.NODES])[0])
    rec("decision_states_ALL", sol.decision_states(to_merge=SolutionMerge.ALL))
    rec("stepwise_states_ALL", sol.stepwise_states(to_merge=SolutionMerge.ALL))
    rec("decision_time_ALL", sol.decision_time(to_merge=SolutionMerge.ALL))
    rec("stepwise_time_ALL", sol.stepwise_time(to_merge=SolutionMerge.ALL))
    rec("decision_controls_ALL", sol.decision_controls(to_merge=SolutionMerge.ALL))
    rec("stepwise_controls_ALL", sol.stepwise_controls(to_merge=SolutionMerge.ALL))
    ip = sol.interpolate(100)
    rec("interpolate100_q", ip["q"])
    # re-simulation with the integrator of the OCP (RK4, same steps), Shooting.SINGLE is the default
    it = sol.integrate(to_merge=[SolutionMerge.KEYS, SolutionMerge.NODES])
    ref = sol.stepwise_states(to_merge=[SolutionMerge.KEYS, SolutionMerge.NODES])
    print("integrate returns", type(it), [np.asarray(a).shape for a in it])
    gap = 0.0
    for p in range(2):
        rec(f"integrate_p{p}", it[p])
        gap = max(gap, float(np.abs(np.asarray(it[p]) - np.asarray(ref[p])).max()))
    res["integrate_gap"] = gap
    print("max |integrate - stepwise_states|", gap)
    res["detailed_cost"] = np.array([float(d["cost_value_weighted"]) for d in sol.detailed_cost])
    rec("decision_time_p0", np.concatenate([np.asarray(a).ravel() for a in sol.decision_time()[0]]))
    rec("decision_time_p1", np.concatenate([np.asarray(a).ravel() for a in sol.decision_time()[1]]))
    rec("stepwise_time_p0", np.concatenate([np.asarray(a).ravel() for a in sol.stepwise_time()[0]]))
    rec("stepwise_time_p1", np.concatenate([np.asarray(a).ravel() for a in sol.stepwise_time()[1]]))
    rec("interpolate100_time", np.linspace(0, sum(TS), 100))
    print("parameters", sol.parameters)
    print("detailed_cost", [(d["name"], d["cost_value_weighted"]) for d in sol.detailed_cost])
    print("cost", sol.cost)
    np.savez(OUT / "solution_pendulum.npz", **res)


if __name__ == "__main__":
    main()
