"""
Real data for ``anim_viz.py``: visualising a solution with ``sol.animate(...)`` (pyorerun / bioviz).

ONE real solve (double pendulum ``double_pendulum.bioMod``, torque driven, N = 30 intervals, T = 1 s, RK4, minimise the
squared torque, q(0) = (0, 0), q(T) = (1.5, -1.0) rad, rest to rest, |tau| <= 30 N.m, IPOPT), then:

* the pyorerun viewer is really run HEADLESSLY: ``sol.animate(viewer="pyorerun", show_now=False, n_frames=200)`` logs the
  animation into rerun's in-memory recording (no window opens) and ``rr.save`` writes it to an ``.rrd`` file (in a
  temporary folder, not in the repository). The file is read back with ``rerun.dataframe`` and the positions of the
  four markers that pyorerun logged at every time stamp are stored: the scene draws its stick figure from them
  (checked against the biorbd markers computed independently from q);
* the same call with the default ``n_frames=0`` is run too, to show that pyorerun ignores ``n_frames`` in this version
  (``animate_with_pyorerun`` does not forward it): 31 time stamps in both cases;
* ``viewer="bioviz"`` is really tried: bioviz is NOT installed here, so the RuntimeError of this version is stored (the
  viewer was NOT run). The frames it would receive are computed with the library's own ``interpolate_data`` (the
  function used by ``viewer_bioviz.py``): 31 frames for ``n_frames=0`` and 200 for ``n_frames=200``.

Output: data/viz_pendulum.npz
Usage (from the repo root):  PYTHONPATH=. python docs/animations/generate_viz_data.py
"""

import os
import tempfile
from pathlib import Path

import biorbd
import numpy as np
import rerun as rr
from bioptim import (
    BoundsList,
    DynamicsOptions,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils
from bioptim.models.biorbd.viewer_bioviz import interpolate_data
from rerun import dataframe as rd

MODEL = ExampleUtils.folder + "/models/double_pendulum.bioMod"
OUT = Path(__file__).parent / "data"
N, T = 30, 1.0
Q_END = [1.5, -1.0]
TAU_MAX = 30.0
RRD_ROOT = "/animation/animation_phase_0/1_double_pendulum"


def prepare_ocp():
    bio_model = TorqueBiorbdModel(MODEL)
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["q"][:, 0] = [0, 0]
    x_bounds["qdot"][:, 0] = [0, 0]
    x_bounds["q"][:, -1] = Q_END
    x_bounds["qdot"][:, -1] = [0, 0]
    u_bounds = BoundsList()
    u_bounds["tau"] = [-TAU_MAX, -TAU_MAX], [TAU_MAX, TAU_MAX]
    return OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=5)),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objectives,
        use_sx=True,
    )


def biorbd_markers(q):
    """(3, n_markers, n_nodes) global marker positions computed by biorbd from q."""
    m = biorbd.Model(MODEL)
    out = np.zeros((3, m.nbMarkers(), q.shape[1]))
    for k in range(q.shape[1]):
        for i, mk in enumerate(m.markers(q[:, k])):
            out[:, i, k] = mk.to_array()
    return out


def animate_headless(sol, path, **kwargs):
    """Run the pyorerun viewer without any window (show_now=False), save the recording, read it back."""
    sol.animate(viewer="pyorerun", show_now=False, **kwargs)
    rr.save(path)  # flushes the in-memory recording to the file
    rec = rd.load_recording(path)
    schema = rec.schema()
    entities = sorted({c.entity_path for c in schema.component_columns()})
    table = rec.view(index="stable_time", contents=RRD_ROOT + "/model_markers/**").select().read_all()
    t = table["stable_time"].to_numpy().astype("datetime64[ns]").astype(np.int64) * 1e-9
    col = [n for n in table.schema.names if n.endswith("model_markers:Position3D")][0]
    pos = np.array(table[col].to_pylist())  # (n_time, 4 markers, 3)
    return t - t[0], pos, len(entities), os.path.getsize(path)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    ocp = prepare_ocp()
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(500)
    sol = ocp.solve(solver)
    stats = sol.ocp.ocp_solver.shaked_ocp_solver.stats()
    print("status", sol.status, "iterations", sol.iterations, "cost", float(sol.cost), stats["return_status"])

    st = sol.decision_states()
    q = np.array([st["q"][k][:, 0] for k in range(N + 1)]).T
    tau = np.array([sol.decision_controls()["tau"][k][:, 0] for k in range(N)]).T
    t = np.concatenate(sol.decision_time()).squeeze()
    mk_biorbd = biorbd_markers(q)
    print("q(T) =", q[:, -1], " max |tau| =", np.abs(tau).max())

    with tempfile.TemporaryDirectory() as tmp:
        t200, pos200, n_ent, size200 = animate_headless(sol, os.path.join(tmp, "n200.rrd"), n_frames=200)
        t0, pos0, _, size0 = animate_headless(sol, os.path.join(tmp, "n0.rrd"))
    print("pyorerun n_frames=200 -> time stamps", len(t200), " default n_frames=0 ->", len(t0))
    print("entities in the recording:", n_ent, " file size (bytes):", size200, size0)
    # rerun logs float32 positions: (x, y, z) of the 4 markers, compare with biorbd
    mk_rrd = np.transpose(pos200, (2, 1, 0))  # (3, 4, 31)
    gap = float(np.abs(mk_rrd - mk_biorbd).max())
    print("max |marker_rrd - marker_biorbd| (m):", gap, " max |t_rrd - t_sol| (s):", float(np.abs(t200 - t).max()))

    try:
        sol.animate(viewer="bioviz", show_now=False, n_frames=200)
        bioviz_msg = "no error (bioviz ran)"
    except Exception as e:  # bioviz is not installed in this environment
        bioviz_msg = f"{type(e).__name__}: {e}"
    print("viewer='bioviz' ->", bioviz_msg)
    n_bioviz_0 = int(interpolate_data(sol, ocp, 0)[0]["q"].shape[1])
    n_bioviz_200 = int(interpolate_data(sol, ocp, 200)[0]["q"].shape[1])
    print("frames handed to bioviz: n_frames=0 ->", n_bioviz_0, " n_frames=200 ->", n_bioviz_200)

    np.savez(
        OUT / "viz_pendulum.npz",
        t=t,
        q=q,
        tau=tau,
        markers_biorbd=mk_biorbd,
        markers_rrd=mk_rrd,
        t_rrd=t200,
        gap_markers=gap,
        n_stamps_200=len(t200),
        n_stamps_0=len(t0),
        n_entities=n_ent,
        rrd_bytes=size200,
        n_bioviz_0=n_bioviz_0,
        n_bioviz_200=n_bioviz_200,
        bioviz_message=bioviz_msg,
        pyorerun_version=__import__("importlib.metadata").metadata.version("pyorerun"),
        rerun_version=rr.__version__,
        status=int(sol.status),
        iterations=int(sol.iterations),
        converged=bool(sol.status == 0),
        cost=float(sol.cost),
        n_shooting=N,
        final_time=T,
        tau_max=TAU_MAX,
    )
    print("saved", OUT / "viz_pendulum.npz")
