"""
Solve the muscle-driven reaching task of bioptim/examples/toy_examples/muscle_driven_ocp/static_arm.py (arm26, 2 dof,
6 muscles) and store REAL IPOPT output in ``data/muscle_arm.npz`` for ``anim_muscle.py``.

Usage (env with bioptim, biorbd, casadi), from the repo root:
    PYTHONPATH=. python docs/animations/generate_muscle_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import Solver, SolutionMerge
from bioptim.examples.toy_examples.muscle_driven_ocp.static_arm import prepare_ocp
from bioptim.examples.utils import ExampleUtils

OUT = Path(__file__).parent / "data"
N, T = 30, 0.5


def main():
    ocp = prepare_ocp(
        biorbd_model_path=ExampleUtils.folder + "/models/arm26_muscle_driven_ocp.bioMod",
        final_time=T,
        n_shooting=N,
        weight=1000,
        n_threads=1,
    )
    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)
    st = sol.decision_states(to_merge=SolutionMerge.NODES)
    ct = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q = np.array(st["q"])
    act = np.array(ct["muscles"])
    tau = np.array(ct["tau"])
    print("status", sol.status, "iters", sol.iterations, "cost", float(sol.cost))
    import biorbd

    m = biorbd.Model(ExampleUtils.folder + "/models/arm26_muscle_driven_ocp.bioMod")
    names = [m.markerNames()[i].to_string() for i in range(m.nbMarkers())]
    marks = {
        n: np.array([m.markers(q[:, k])[names.index(n)].to_array() for k in range(N + 1)])
        for n in ("r_acromion", "r_humerus_epicondyle", "COM_hand", "target")
    }
    err = float(np.linalg.norm(marks["target"][-1] - marks["COM_hand"][-1]))
    print("marker error [m]", err)
    np.savez(
        OUT / "muscle_arm.npz",
        t=np.linspace(0, T, N + 1),
        q=q,
        act=act,
        tau=tau,
        muscle_names=np.array([m.muscleNames()[i].to_string() for i in range(m.nbMuscles())]),
        marker_error=err,
        shoulder=marks["r_acromion"],
        elbow=marks["r_humerus_epicondyle"],
        hand=marks["COM_hand"],
        target=marks["target"],
        cost=float(sol.cost),
        iterations=int(sol.iterations),
        converged=int(sol.status == 0),
    )


if __name__ == "__main__":
    main()
