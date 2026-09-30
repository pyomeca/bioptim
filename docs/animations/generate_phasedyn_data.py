"""
PhaseDynamics.SHARED_DURING_THE_PHASE vs PhaseDynamics.ONE_PER_NODE, measured on REAL bioptim OCPs.

Problem: the time-dependent pendulum of ``examples/toy_examples/torque_driven_ocp/example_pendulum_time_dependent.py``
(same model class TimeDependentModel, RK4, SX, minimise tau, swing to 3.14 rad, N nodes, T = 1 s), rebuilt here only to
be able to add a multinode constraint.  Everything stored in ``data/phasedyn_*.npz`` is measured:

    phasedyn_bench.npz   for N in (30, 60, 120) and both options: median of 3 (min/max kept) of the OCP build time and of
                         the IPOPT solve time, number of DISTINCT integrator casadi Functions in nlp.dynamics, total
                         number of SX nodes of the constraint vector g, iterations, cost, q trajectory (N = 30).
    phasedyn_hold.npz    the case where ONE_PER_NODE is required: a 4-node multinode constraint in a single phase
                         (the rotation equal at nodes 24 to 30, one CUSTOM multinode constraint on 7 nodes).  SHARED raises ValueError (message stored), ONE_PER_NODE
                         solves; the unconstrained solution is kept as a ghost.
    phasedyn_series.npz  numerical_data_timeseries (external forces, examples/getting_started/example_external_forces.py):
                         both options solve and reach the same cost, i.e. a time-varying force alone does NOT require
                         ONE_PER_NODE in this version.

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_phasedyn_data.py
"""

import time
from pathlib import Path

import numpy as np
from casadi import Function, vertcat
from bioptim import (
    BoundsList,
    DynamicsOptions,
    InitialGuessList,
    MultinodeConstraintFcn,
    MultinodeConstraintList,
    Objective,
    ObjectiveFcn,
    OdeSolver,
    OptimalControlProgram,
    PhaseDynamics,
    Solver,
    SolutionMerge,
)
from bioptim.examples.getting_started import example_external_forces as ext_ex
from bioptim.examples.toy_examples.torque_driven_ocp.example_pendulum_time_dependent import TimeDependentModel
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
T = 1.0
SHARED, PER_NODE = PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE
OPTIONS = {"shared": SHARED, "per_node": PER_NODE}
HOLD_NODES = (24, 25, 26, 27, 28, 29, 30)


def hold_rotation(controllers):
    """One multinode constraint on len(controllers) nodes: the rotation must equal that of the first node."""
    q_first = controllers[0].states["q"].cx[1]
    return vertcat(*[q_first - c.states["q"].cx[1] for c in controllers[1:]])


def prepare_ocp(n_shooting, phase_dynamics, hold=False):
    bio_model = TimeDependentModel(MODEL)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 3.14
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * bio_model.nb_tau, [100] * bio_model.nb_tau
    u_bounds["tau"][1, :] = 0
    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model.nb_q
    x_init["qdot"] = [0] * bio_model.nb_qdot
    multinode = MultinodeConstraintList()
    if hold:
        multinode.add(
            MultinodeConstraintFcn.CUSTOM,
            custom_function=hold_rotation,
            nodes_phase=(0,) * len(HOLD_NODES),
            nodes=HOLD_NODES,
        )
    return OptimalControlProgram(
        bio_model,
        n_shooting,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(), phase_dynamics=phase_dynamics),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"),
        multinode_constraints=multinode,
        use_sx=True,
    )


def solve(ocp):
    solver = Solver.IPOPT()
    solver.set_print_level(0)
    t0 = time.perf_counter()
    sol = ocp.solve(solver)
    return sol, time.perf_counter() - t0


def n_unique_integrators(ocp):
    return len({id(f) for f in ocp.nlp[0].dynamics})


def q_of(sol):
    return sol.decision_states(to_merge=SolutionMerge.NODES)["q"]


def bench():
    res = {}
    for n in (30, 60, 120):
        for name, pd in OPTIONS.items():
            builds, solves = [], []
            for _ in range(3):
                t0 = time.perf_counter()
                ocp = prepare_ocp(n, pd)
                builds.append(time.perf_counter() - t0)
                sol, ts = solve(ocp)
                solves.append(ts)
            g = ocp.ocp_solver.nlp["g"]
            graph = Function("g", [ocp.ocp_solver.nlp["x"]], [g]).n_nodes()
            key = f"{name}_{n}"
            res[f"{key}_build"], res[f"{key}_solve"] = np.array(builds), np.array(solves)
            res[f"{key}_unique_integrators"] = n_unique_integrators(ocp)
            res[f"{key}_graph_nodes"] = graph
            res[f"{key}_iterations"], res[f"{key}_cost"], res[f"{key}_status"] = (
                sol.iterations,
                float(sol.cost),
                sol.status,
            )
            if n == 30:
                res[f"{name}_q"] = q_of(sol)
            print(
                key,
                "build",
                np.round(builds, 2),
                "solve",
                np.round(solves, 2),
                "unique",
                res[f"{key}_unique_integrators"],
                "graph",
                graph,
                "it",
                sol.iterations,
                "cost",
                float(sol.cost),
                "status",
                sol.status,
            )
    res["max_dq"] = float(np.abs(res["shared_q"] - res["per_node_q"]).max())
    print("max |q_shared - q_per_node| =", res["max_dq"])
    np.savez(OUT / "phasedyn_bench.npz", **res)


def hold():
    res = {"nodes": np.array(HOLD_NODES), "T": T, "N": 30}
    free_sol, _ = solve(prepare_ocp(30, PER_NODE))
    res["free_q"] = q_of(free_sol)
    res["free_cost"] = float(free_sol.cost)
    try:
        prepare_ocp(30, SHARED, hold=True)
        res["shared_error"] = "no error"
    except Exception as e:
        res["shared_error"] = f"{type(e).__name__}: {e}"
    print("SHARED ->", res["shared_error"])
    ocp = prepare_ocp(30, PER_NODE, hold=True)
    sol, _ = solve(ocp)
    res["hold_q"], res["hold_cost"], res["hold_status"], res["hold_iterations"] = (
        q_of(sol),
        float(sol.cost),
        sol.status,
        sol.iterations,
    )
    print(
        "ONE_PER_NODE hold: status",
        sol.status,
        "iterations",
        sol.iterations,
        "cost",
        float(sol.cost),
        "free",
        res["free_cost"],
    )
    print("q_rot at hold nodes", res["hold_q"][1, list(HOLD_NODES)], "free", res["free_q"][1, list(HOLD_NODES)])
    np.savez(OUT / "phasedyn_hold.npz", **res)


def series():
    res = {}
    model = ExampleUtils.folder + "/models/cube_with_forces.bioMod"
    for name, pd in OPTIONS.items():
        ocp = ext_ex.prepare_ocp(model, phase_dynamics=pd, use_sx=False)
        sol, _ = solve(ocp)
        res[f"{name}_cost"], res[f"{name}_status"] = float(sol.cost), sol.status
        res[f"{name}_q"] = q_of(sol)
        print("external forces", name, "status", sol.status, "cost", float(sol.cost))
    res["max_dq"] = float(np.abs(res["shared_q"] - res["per_node_q"]).max())
    print("external forces max |dq|", res["max_dq"])
    np.savez(OUT / "phasedyn_series.npz", **res)


if __name__ == "__main__":
    hold()
    series()
    bench()
