"""
Penalty on the derivative of a control: ``Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True)``.
Everything stored in ``data/deriv_pendulum.npz`` is REAL bioptim / IPOPT output (cart-pendulum swing-up, model of
generate_features_data.py, N = 30 intervals, T = 1 s, RK4, rotation from 0 to 3.14 rad at rest, only the sliding
translation is actuated).

What ``derivative=True`` does in THIS version (bioptim/limits/penalty_option.py, branch ``elif self.derivative``): the
penalty function is evaluated at the end and at the start of the interval and subtracted, f(u_end) - f(u_start), with
    u_end = u_start                      if the control type is CONSTANT or CONSTANT_WITH_LAST_NODE   (=> always 0!)
    u_end = controls.cx_end              otherwise (LINEAR_CONTINUOUS)
so with the default ``ControlType.CONSTANT`` the term is identically zero and the solution does not change (checked
below, ``const_*`` keys). It becomes meaningful with ``control_type=ControlType.LINEAR_CONTINUOUS``. (``explicit_derivative``
is not a user option in practice: it is set internally by the continuity constraints.)

Runs (LINEAR_CONTINUOUS, weight of the derivative term w, plain MINIMIZE_CONTROL weight 1): w = 0, 1, 10, 100,
solved by continuation (each solve is warm started by the previous one). Two CONSTANT runs (w = 0 and 100) show that
derivative=True changes nothing there.

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_deriv_data.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bioptim import (
    ControlType,
    InitialGuessList,
    InterpolationType,
    Objective,
    ObjectiveFcn,
    ObjectiveList,
    OptimalControlProgram,
    Solver,
    TorqueBiorbdModel,
)
from generate_features_data import MODEL, ROT, pendulum_bounds, rk4

OUT = Path(__file__).parent / "data"
N, T = 30, 1.0
WEIGHTS = [0.0, 1.0, 10.0, 100.0]


def prepare_ocp(w, control_type, previous=None):
    bio_model = TorqueBiorbdModel(MODEL)
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1.0)
    if w > 0:
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=w)
    x_bounds, u_bounds = pendulum_bounds(bio_model)
    x_init, u_init = InitialGuessList(), InitialGuessList()
    if previous is not None:
        each = InterpolationType.EACH_FRAME
        st, co = previous
        x_init.add("q", st["q"], interpolation=each)
        x_init.add("qdot", st["qdot"], interpolation=each)
        u_init.add("tau", co, interpolation=each)
    return OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=rk4(),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objectives,
        control_type=control_type,
        use_sx=True,
    )


def run(w, control_type, previous=None):
    ocp = prepare_ocp(w, control_type, previous)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver)
    st, co = sol.decision_states(), sol.decision_controls()
    q = np.array([st["q"][k][:, 0] for k in range(N + 1)]).T
    qdot = np.array([st["qdot"][k][:, 0] for k in range(N + 1)]).T
    tau_full = np.array([co["tau"][k][:, 0] for k in range(len(co["tau"]))]).T
    return sol, dict(q=q, qdot=qdot, tau=tau_full[0], tau_full=tau_full)


def main():
    res, prev = {}, None
    for w in WEIGHTS:
        sol, d = run(w, ControlType.LINEAR_CONTINUOUS, None if prev is None else (prev["st"], prev["tau"]))
        prev = dict(st=dict(q=d["q"], qdot=d["qdot"]), tau=d["tau_full"])
        tau = d["tau"]
        dt = T / N
        dtau = np.diff(tau) / dt
        res[f"w{int(w)}"] = dict(
            w=w,
            tau=tau,
            q_rot=d["q"][ROT],
            qdot_rot=d["qdot"][ROT],
            cost=float(sol.cost),
            status=int(sol.status),
            iterations=int(sol.iterations),
            effort=float(np.sum((tau[:-1] ** 2 + tau[:-1] * tau[1:] + tau[1:] ** 2) / 3) * dt),  # exact int tau^2 dt
            dtau_max=float(np.abs(dtau).max()),
            dtau_rms=float(np.sqrt(np.mean(dtau**2))),
            tau_max=float(np.abs(tau).max()),
            q_end=float(d["q"][ROT, -1]),
        )
        r = res[f"w{int(w)}"]
        print(
            f"LINEAR w={w}: status={r['status']} it={r['iterations']} cost={r['cost']:.3f} effort={r['effort']:.1f} "
            f"max|dtau/dt|={r['dtau_max']:.1f} rms={r['dtau_rms']:.1f} max|tau|={r['tau_max']:.1f} q_end={r['q_end']:.3f}"
        )
    const = {}
    for w in (0.0, 100.0):
        sol, d = run(w, ControlType.CONSTANT)
        const[f"const_w{int(w)}"] = dict(cost=float(sol.cost), status=int(sol.status), tau=d["tau"])
        print(f"CONSTANT w={w}: status={sol.status} cost={float(sol.cost):.6f} max|tau|={np.abs(d['tau']).max():.2f}")
    flat = {f"{tag}_{k}": v for tag, d in {**res, **const}.items() for k, v in d.items()}
    flat["N"], flat["T"] = N, T
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "deriv_pendulum.npz", **flat)


if __name__ == "__main__":
    main()
