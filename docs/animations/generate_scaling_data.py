"""
Solve the pendulum OCP of bioptim/examples/toy_examples/feature_examples/example_variable_scaling.py twice with REAL
bioptim / IPOPT solves, to feed ``anim_scaling.py``:
    - "unscaled": same problem, all scaling factors set to 1
    - "scaled"  : x_scaling q=[1, 3], qdot=[85, 85], u_scaling tau=[900, 1]  (the example's values)
The IPOPT iteration history (objective, inf_pr, inf_du) is parsed from IPOPT's own output file. IPOPT's internal
gradient-based nlp scaling stays at its default (on) for both runs.
Output: data/scaling_pendulum.npz

Usage (env with bioptim on the PATH), from the repo root:
    PYTHONPATH=. python docs/animations/generate_scaling_data.py
"""

import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from bioptim import Solver, SolutionMerge
from bioptim.examples.toy_examples.feature_examples import example_variable_scaling as ex
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
T, N = 0.1, 30  # same as example_variable_scaling.main()
X_SC = {"q": [1.0, 3.0], "qdot": [85.0, 85.0]}
U_SC = {"tau": [900.0, 1.0]}
ITER = re.compile(
    r"^\s*(\d+)r?\s+(\S+)\s+(\S+)\s+(\S+)\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\d+\s*$"
)  # iter obj inf_pr inf_du lg(mu) ||d|| lg(rg) alpha_du alpha_pr ls


def build(scaled):
    original = ex.VariableScalingList
    if not scaled:

        class Unit(original):  # all factors = 1, the rest of the OCP is untouched
            def __setitem__(self, key, value):
                super().__setitem__(key, [1.0] * len(value))

        ex.VariableScalingList = Unit
    try:
        return ex.prepare_ocp(MODEL, T, N)
    finally:
        ex.VariableScalingList = original


def _solve_child(scaled, nlp_scaling_method, log):
    """Runs in a child process: on Windows IPOPT keeps its output file locked until the process exits."""
    ocp = build(scaled)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(1000)
    solver.set_nlp_scaling_method(nlp_scaling_method)
    solver.set_option_unsafe(str(log), "output_file")
    solver.set_option_unsafe(5, "file_print_level")
    sol = ocp.solve(solver)
    st, ct = sol.decision_states(to_merge=SolutionMerge.NODES), sol.decision_controls(to_merge=SolutionMerge.NODES)
    return dict(
        iterations=int(sol.iterations),
        status=int(sol.status),
        cost=float(sol.cost),
        q=st["q"],
        qdot=st["qdot"],
        tau=ct["tau"],
    )


def solve(scaled, nlp_scaling_method="gradient-based"):
    log = Path(tempfile.mkdtemp()) / "ipopt.txt"
    with ProcessPoolExecutor(max_workers=1) as pool:
        res = pool.submit(_solve_child, scaled, nlp_scaling_method, log).result()
    with open(log) as f:
        rows = [ITER.match(line) for line in f.read().splitlines()]
    res["hist"] = np.array([[float(m.group(i)) for i in (1, 2, 3, 4)] for m in rows if m])
    return res


if __name__ == "__main__":
    out = {"n_shooting": N, "final_time": T}
    for tag, scaled in (("unscaled", False), ("scaled", True)):
        r = solve(scaled)
        for k in ("hist", "iterations", "status", "cost", "q", "qdot", "tau"):
            out[f"{tag}_{k}"] = r[k]
        out[f"{tag}_t"] = np.linspace(0, T, N + 1)
        print(tag, "status", r["status"], "iters", r["iterations"], "cost", r["cost"], "rows", len(r["hist"]))
    # IPOPT's own nlp_scaling_method = "none" (robustness check quoted in the notes)
    for tag, scaled in (("unscaled", False), ("scaled", True)):
        r = solve(scaled, "none")
        out[f"{tag}_iterations_ipopt_none"] = r["iterations"]
        print(tag, "IPOPT nlp_scaling none: iters", r["iterations"], "status", r["status"], "cost", r["cost"])
    for k, v in {**X_SC, **U_SC}.items():
        out[f"factor_{k}"] = np.array(v)
    np.savez(OUT / "scaling_pendulum.npz", **out)
