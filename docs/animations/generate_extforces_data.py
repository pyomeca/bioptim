"""
External forces: a wind-like push on the hand of a planar two-link arm (models/extforces_arm.bioMod), passed as an
``ExternalForceSetTimeSeries`` + ``numerical_data_timeseries``. Two REAL bioptim / IPOPT solves of the same reaching
movement (N = 30, T = 1.5 s, minimise tau): without the force (ghost) and with it. Stored in ``data/extforces_arm.npz``.
Stick figures come from biorbd marker positions (elbow, hand) computed from the optimal q.

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_extforces_data.py
"""

from pathlib import Path

import numpy as np
from casadi import MX
from bioptim import (
    BoundsList,
    DynamicsOptions,
    ExternalForceSetTimeSeries,
    InitialGuessList,
    InterpolationType,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    Node,
    SolutionMerge,
    TorqueBiorbdModel,
)

HERE = Path(__file__).parent
MODEL = str(HERE / "models" / "extforces_arm.bioMod")
OUT = HERE / "data"
N, T = 30, 1.5
Q_END = np.array([1.2, 0.9])
PEAK = 15.0  # N
W_TRACK = 100.0


def external_force(n_shooting):
    """Wind-like push along -y on the hand: a smooth bump centred at 60 % of the movement."""
    t = np.linspace(0, T, n_shooting + 1)[:-1]
    bump = PEAK * np.exp(-(((t - 0.6 * T) / (0.2 * T)) ** 2))
    force = np.zeros((3, n_shooting))
    force[1, :] = -bump
    fset = ExternalForceSetTimeSeries(nb_frames=n_shooting)
    hand = np.tile([[0], [0], [-0.3]], (1, n_shooting))  # the hand marker, in the Forearm frame
    fset.add_translational_force("push", "Forearm", force, point_of_application_in_local=hand)
    return fset, force


def reference():
    """Smooth (minimum-jerk) reach from q = 0 to Q_END, sampled at the N + 1 nodes."""
    s = np.linspace(0, 1, N + 1)
    return Q_END[:, None] * (10 * s**3 - 15 * s**4 + 6 * s**5)[None, :]


def prepare_ocp(with_force, x_init=None, u_init=None):
    fset, _ = external_force(N)
    bio_model = TorqueBiorbdModel(MODEL, external_force_set=fset)
    dynamics = (
        DynamicsOptions(
            ode_solver=OdeSolver.RK4(),
            numerical_data_timeseries={"external_forces": fset.to_numerical_time_series()},
        )
        if with_force
        else DynamicsOptions(ode_solver=OdeSolver.RK4())
    )
    if not with_force:
        bio_model = TorqueBiorbdModel(MODEL)
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objectives.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=W_TRACK, target=reference(), node=Node.ALL)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0
    x_bounds["q"][:, -1] = Q_END
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * bio_model.nb_tau, [100] * bio_model.nb_tau
    xi = InitialGuessList()
    xi.add("q", np.array([[0, Q_END[0]], [0, Q_END[1]]]), interpolation=InterpolationType.LINEAR)
    return OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=xi,
        objective_functions=objectives,
        use_sx=False,
    )


def solve(with_force):
    ocp = prepare_ocp(with_force)
    solver = Solver.IPOPT()
    solver.set_print_level(0)
    solver.set_maximum_iterations(500)
    sol = ocp.solve(solver)
    st = sol.decision_states(to_merge=SolutionMerge.NODES)
    ct = sol.decision_controls(to_merge=SolutionMerge.NODES)
    return ocp, sol, st["q"], st["qdot"], ct["tau"]


def main():
    res = {}
    _, force = external_force(N)
    for name, wf in (("free", False), ("push", True)):
        ocp, sol, q, qd, tau = solve(wf)
        print(
            name, "status", sol.status, "iters", sol.iterations, "cost", float(sol.cost), "tau", tau.shape, "q", q.shape
        )
        res[f"{name}_q"], res[f"{name}_qdot"], res[f"{name}_tau"] = q, qd, tau
        res[f"{name}_status"], res[f"{name}_iterations"], res[f"{name}_cost"] = (
            sol.status,
            sol.iterations,
            float(sol.cost),
        )
        # stick figure from biorbd markers
        bm = ocp.nlp[0].model
        names = list(bm.marker_names)
        funcs = {n_: bm.marker(names.index(n_)) for n_ in ("elbow", "hand")}
        for n_, f in funcs.items():
            res[f"{name}_{n_}"] = np.array([np.array(f(q[:, k], [])).ravel() for k in range(q.shape[1])])
    res.update(reference=reference(), force=force, T=T, N=N, peak=PEAK, q_end=Q_END)
    np.savez(OUT / "extforces_arm.npz", **res)
    dt = res["push_tau"][:, :N] - res["free_tau"][:, :N]
    print(
        "max |dtau|",
        np.abs(dt).max(axis=1),
        "tau free max",
        np.abs(res["free_tau"]).max(axis=1),
        "push",
        np.abs(res["push_tau"]).max(axis=1),
    )
    print("hand push y at k=18", res["push_hand"][18], res["free_hand"][18])


if __name__ == "__main__":
    main()
