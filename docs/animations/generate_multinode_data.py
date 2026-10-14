"""
Multinode constraint / objective: link the FIRST and LAST node of a cart-pendulum movement (cyclic movement).
Everything stored in ``data/multinode_pendulum.npz`` is REAL bioptim / IPOPT output (pendulum.bioMod, N = 30, T = 2 s,
sliding translation actuated, passive rotation). Start: q = 0 (qdot free). Middle node (Node.MID): rotation = 1 rad.
Minimise the integral of tau^2.  Three solves, only the link between Node.START and Node.END changes:
    free : no link
    obj  : MultinodeObjectiveFcn.STATES_EQUALITY (soft, weight W_OBJ)
    cons : MultinodeConstraintFcn.STATES_EQUALITY (hard)

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_multinode_data.py
"""

import time
from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    ConstraintFcn,
    ConstraintList,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    MultinodeConstraintFcn,
    MultinodeConstraintList,
    MultinodeObjectiveFcn,
    MultinodeObjectiveList,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
ROT, T, N = 1, 2.0, 30
W_OBJ = 10.0


def prepare_ocp(mode):
    bio_model = TorqueBiorbdModel(MODEL)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * bio_model.nb_tau, [100] * bio_model.nb_tau
    u_bounds["tau"][ROT, :] = 0
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", index=ROT, node=Node.MID, target=1.0)
    mn_cons, mn_obj = MultinodeConstraintList(), MultinodeObjectiveList()
    if mode == "cons":
        mn_cons.add(MultinodeConstraintFcn.STATES_EQUALITY, nodes_phase=(0, 0), nodes=(Node.START, Node.END), key="all")
    elif mode == "obj":
        mn_obj.add(
            MultinodeObjectiveFcn.STATES_EQUALITY,
            nodes_phase=(0, 0),
            nodes=(Node.START, Node.END),
            key="all",
            weight=W_OBJ,
        )
    return OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4()),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objectives,
        constraints=constraints,
        multinode_constraints=mn_cons,
        multinode_objectives=mn_obj,
        use_sx=True,
    )


def main():
    res = {}
    for mode in ["free", "obj", "cons"]:
        ocp = prepare_ocp(mode)
        solver = Solver.IPOPT(show_online_optim=False)
        solver.set_print_level(0)
        solver.set_maximum_iterations(1000)
        t0 = time.time()
        sol = ocp.solve(solver)
        st, ct = sol.decision_states(), sol.decision_controls()
        x = np.array([np.concatenate([st["q"][k][:, 0], st["qdot"][k][:, 0]]) for k in range(N + 1)]).T
        tau = np.array([ct["tau"][k][:, 0] for k in range(N)]).T
        gap = x[:, -1] - x[:, 0]
        res[mode] = dict(
            x=x,
            tau=tau,
            gap=gap,
            gap_norm=float(np.linalg.norm(gap)),
            cost=float(sol.cost),
            status=int(sol.status),
            iterations=int(sol.iterations),
        )
        print(
            f"{mode}: status={sol.status} it={sol.iterations} cost={res[mode]['cost']:.4f} |gap|={res[mode]['gap_norm']:.2e} "
            f"gap={np.round(gap, 4)} t={time.time() - t0:.1f}s"
        )
    flat = {f"{m}_{k}": v for m, d in res.items() for k, v in d.items()}
    flat.update(N=N, T=T, w_obj=W_OBJ)
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "multinode_pendulum.npz", **flat)


if __name__ == "__main__":
    main()
