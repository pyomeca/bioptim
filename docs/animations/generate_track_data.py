"""
Track a smooth reference pole angle with ObjectiveFcn.Lagrange.TRACK_STATE for three weights (REAL bioptim / IPOPT
solves, warm started from the previous weight) to feed ``anim_track.py``. Also solves once with the hard-constraint
version ConstraintFcn.TRACK_STATE for comparison. Output: data/track_pendulum.npz

Usage (env with bioptim on the PATH), from the repo root:
    PYTHONPATH=. python docs/animations/generate_track_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    ConstraintFcn,
    ConstraintList,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    SolutionMerge,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
ROT, N, T, TAU_MAX = 1, 40, 2.0, 100.0
WEIGHTS = [1.0, 30.0, 1000.0]
W_CTRL = 1.0
AMP = 0.3


def reference(t):
    """Smooth pole angle: one period of a sine, amplitude AMP (rad)."""
    return AMP * np.sin(2 * np.pi * t / T)


def build(weight, hard=False):
    bio_model = TorqueBiorbdModel(MODEL)
    t_nodes = np.linspace(0, T, N + 1)
    target = reference(t_nodes)[None, :]
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=W_CTRL, node=Node.ALL_SHOOTING)
    constraints = ConstraintList()
    if hard:
        constraints.add(ConstraintFcn.TRACK_STATE, key="q", index=[ROT], node=Node.ALL, target=target)
    else:
        objectives.add(
            ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=[ROT], weight=weight, node=Node.ALL, target=target
        )
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["q"].min[0, :] = -4
    x_bounds["q"].max[0, :] = 4
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"][:, 0] = 0  # starts at rest (the reference starts with a non-zero angular velocity)
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
        objective_functions=objectives,
        constraints=constraints,
        use_sx=True,
    )
    return ocp


def solve(ocp, warm=None):
    if warm is not None:
        x_init = InitialGuessList()
        x_init.add("q", warm["q"], interpolation=InterpolationType.EACH_FRAME)
        x_init.add("qdot", warm["qdot"], interpolation=InterpolationType.EACH_FRAME)
        u_init = InitialGuessList()
        u_init.add("tau", warm["tau"], interpolation=InterpolationType.EACH_FRAME)
        ocp.update_initial_guess(x_init, u_init)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(1000)
    return ocp.solve(solver)


def extract(sol):
    st, ct = sol.decision_states(), sol.decision_controls()
    q = np.array([st["q"][k][:, 0] for k in range(N + 1)]).T
    qdot = np.array([st["qdot"][k][:, 0] for k in range(N + 1)]).T
    tau = np.array([ct["tau"][k][:, 0] for k in range(len(ct["tau"]))]).T
    return dict(q=q, qdot=qdot, tau=tau)


if __name__ == "__main__":
    t = np.linspace(0, T, N + 1)
    out = {"n_shooting": N, "final_time": T, "weights": np.array(WEIGHTS), "w_ctrl": W_CTRL, "t": t}
    out["target"] = reference(t)
    dt = T / N
    warm = None
    runs = [("w1", WEIGHTS[1], False), ("w2", WEIGHTS[2], False), ("hard", 0.0, True), ("w0", WEIGHTS[0], False)]
    for tag, w, hard in runs:
        ocp = build(w, hard)
        sol = solve(ocp, warm)
        r = extract(sol)
        if tag in ("w1", "w2"):
            warm = r  # continuation: 10 -> 1000 -> hard; w0 restarts from the weight-10 solution
        if tag == "w1":
            w1_sol = r
        if tag == "hard":
            warm = w1_sol
        err = r["q"][ROT] - out["target"]
        tau0 = r["tau"][0]
        out[f"{tag}_theta"] = r["q"][ROT]
        out[f"{tag}_cart"] = r["q"][0]
        out[f"{tag}_tau"] = r["tau"][0]
        out[f"{tag}_err"] = err
        out[f"{tag}_rms_err"] = float(np.sqrt(np.mean(err**2)))
        out[f"{tag}_max_err"] = float(np.max(np.abs(err)))
        out[f"{tag}_effort"] = float(np.sum(tau0**2) * dt)  # ~ integral of tau^2 dt
        out[f"{tag}_cost"] = float(sol.cost)
        out[f"{tag}_iterations"] = int(sol.iterations)
        out[f"{tag}_status"] = int(sol.status)
        print(
            tag,
            w,
            "status",
            sol.status,
            "iters",
            sol.iterations,
            "rms",
            out[f"{tag}_rms_err"],
            "max",
            out[f"{tag}_max_err"],
            "effort",
            out[f"{tag}_effort"],
            "max|tau|",
            np.abs(tau0).max(),
            "cart range",
            r["q"][0].min(),
            r["q"][0].max(),
        )
    np.savez(OUT / "track_pendulum.npz", **out)
