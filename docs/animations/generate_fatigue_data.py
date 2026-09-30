"""
Solve the pendulum swing-up with a fatigue model on the torques (REAL bioptim / IPOPT solve) to feed ``anim_fatigue.py``.
Uses ``prepare_ocp`` of bioptim/examples/toy_examples/fatigue/pendulum_with_fatigue.py (fatigue_type="xia").
Output: data/fatigue_xia.npz

Usage (env with bioptim on the PATH), from the repo root:
    PYTHONPATH=. python docs/animations/generate_fatigue_data.py
"""

import sys
from pathlib import Path

import numpy as np
from bioptim import Solver, SolutionMerge
from bioptim.examples.utils import ExampleUtils
from bioptim.examples.toy_examples.fatigue.pendulum_with_fatigue import prepare_ocp

OUT = Path(__file__).parent / "data"
N = 30
T = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

if __name__ == "__main__":
    ocp = prepare_ocp(
        ExampleUtils.folder + "/models/pendulum.bioMod",
        final_time=T,
        n_shooting=N,
        fatigue_type="xia",
        split_controls=False,
    )
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver)
    st = sol.decision_states(to_merge=SolutionMerge.NODES)
    ct = sol.decision_controls()
    print("status", sol.status, "iters", sol.iterations, "cost", sol.cost, "keys", list(st.keys()))
    out = {"n_shooting": N, "final_time": T, "status": sol.status, "iterations": sol.iterations, "cost": sol.cost}
    for k in st:
        out[k] = np.array(st[k])
        print(k, np.array(st[k]).shape)
    tau = np.array([ct["tau"][k][:, 0] for k in range(len(ct["tau"]))]).T
    out["tau"] = tau
    print("tau", tau.shape, np.round(tau[0], 1))
    np.savez(OUT / "fatigue_xia.npz", **out)
