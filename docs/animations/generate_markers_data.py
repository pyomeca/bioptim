"""
Real IPOPT data for ``anim_markers.py``: tracking measured markers with ObjectiveFcn.Lagrange.TRACK_MARKERS.

"Measured" markers are SYNTHETIC: a known joint motion q_true(t) of the double pendulum
(bioptim/examples/models/double_pendulum.bioMod, planar in the y-z plane) is passed through the model's marker
function and Gaussian noise (sigma = 1 cm) is added. The OCP (N = 30, T = 1.5 s, free initial/final states, torque
driven) then tracks markers marker_2 (elbow) and marker_4 (tip) in the plane. The trajectory after k IPOPT iterations is
obtained by re-solving with ``set_maximum_iterations(k)`` (IPOPT is deterministic).

Output: data/markers_tracking.npz
Usage (from the repo root):  PYTHONPATH=. python docs/animations/generate_markers_data.py
"""

from pathlib import Path

import biorbd
import numpy as np
from bioptim import (
    Axis,
    BoundsList,
    DynamicsOptions,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/double_pendulum.bioMod"
OUT = Path(__file__).parent / "data"
N, T = 30, 1.5
SIGMA = 0.01  # marker noise (m)
TRACKED = [1, 3]  # marker_2 (elbow), marker_4 (tip)
ITER_LIST = [0, 1, 2, 4, 6, 8, 10]


def true_motion():
    t = np.linspace(0, T, N + 1)
    q = np.vstack((1.0 * np.sin(2 * np.pi * t / T), 0.6 * (1 - np.cos(2 * np.pi * t / T))))
    return t, q


def markers_of(q):
    """(3, n_markers, n_nodes) global marker positions from biorbd."""
    m = biorbd.Model(MODEL)
    out = np.zeros((3, m.nbMarkers(), q.shape[1]))
    for k in range(q.shape[1]):
        for i, mk in enumerate(m.markers(q[:, k])):
            out[:, i, k] = mk.to_array()
    return out


def build(target):
    bio_model = TorqueBiorbdModel(MODEL)
    objectives = ObjectiveList()
    objectives.add(
        ObjectiveFcn.Lagrange.TRACK_MARKERS,
        weight=1000,
        node=Node.ALL,
        marker_index=TRACKED,
        axes=[Axis.Y, Axis.Z],
        target=target,
    )
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1e-3, node=Node.ALL_SHOOTING)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    return OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=5)),
        x_bounds=x_bounds,
        objective_functions=objectives,
        use_sx=True,
    )


def solve(ocp, max_iter=500):
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(max_iter)
    return ocp.solve(solver)


def q_of(sol):
    st = sol.decision_states()
    return np.array([st["q"][k][:, 0] for k in range(N + 1)]).T


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    t, q_true = true_motion()
    clean = markers_of(q_true)
    rng = np.random.default_rng(0)
    noisy = clean + rng.normal(0, SIGMA, clean.shape)
    noisy[0] = 0  # planar: x is not tracked
    target = noisy[1:, TRACKED, :]  # (2 axes y-z, 2 markers, N+1)

    ocp = build(target)
    out = {
        "t": t,
        "q_true": q_true,
        "markers_clean": clean,
        "markers_meas": noisy,
        "tracked": np.array(TRACKED),
        "sigma": SIGMA,
        "n_shooting": N,
        "final_time": T,
        "iter_list": np.array(ITER_LIST),
    }
    for k in ITER_LIST:
        s = solve(build(target), k)
        q = q_of(s)
        out[f"q_it{k}"] = q
        out[f"markers_it{k}"] = markers_of(q)
        print("iter", k, "status", s.status, "cost", s.cost)
    sol = solve(ocp)
    stats = sol.ocp.ocp_solver.shaked_ocp_solver.stats()
    q = q_of(sol)
    out["q_opt"] = q
    out["markers_opt"] = markers_of(q)
    out["status"] = sol.status
    out["iterations"] = int(sol.iterations)
    out["cost"] = float(sol.cost)
    out["tau"] = np.array([sol.decision_controls()["tau"][k][:, 0] for k in range(N)]).T
    print("final status", sol.status, "iters", sol.iterations, "cost", sol.cost, stats["return_status"])

    def err(mk):
        d = mk[1:, TRACKED, :] - noisy[1:, TRACKED, :]
        return np.linalg.norm(d, axis=0)  # (2 markers, N+1)

    e = err(out["markers_opt"])
    print("rms marker error vs measured (cm):", 100 * np.sqrt((e**2).mean()))
    ec = np.linalg.norm(out["markers_opt"][1:, TRACKED, :] - clean[1:, TRACKED, :], axis=0)
    print("rms marker error vs clean truth (cm):", 100 * np.sqrt((ec**2).mean()))
    en = np.linalg.norm(noisy[1:, TRACKED, :] - clean[1:, TRACKED, :], axis=0)
    print("rms noise (cm):", 100 * np.sqrt((en**2).mean()))
    print("rms q error (deg):", np.degrees(np.sqrt(((q - q_true) ** 2).mean())))
    print("q at last node vs true:", q[:, -1], q_true[:, -1])
    np.savez(OUT / "markers_tracking.npz", **out)
