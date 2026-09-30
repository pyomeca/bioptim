"""
Solve the bioptim holonomic example ``two_pendulums`` (two single pendulums glued by a holonomic constraint into a
double pendulum) and store the REAL result in ``data/holonomic_two_pendulums.npz`` for ``anim_holonomic.py``.

q = (theta0, y1, z1, theta1): joints 0 and 3 (rotations) are independent (u), joints 1 and 2 (translations of the
second segment) are dependent (v), recovered from the constraint marker_1 (tip of segment 0) == marker_3 (origin of segment 1).

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_holonomic_data.py
"""

from pathlib import Path

import biorbd
import numpy as np
from bioptim import SolutionMerge, Solver
from bioptim.examples.toy_examples.holonomic_constraints.two_pendulums import prepare_ocp
from bioptim.examples.utils import ExampleUtils

OUT = Path(__file__).parent / "data"
N_SHOOTING = 30


def main():
    path = ExampleUtils.folder + "/models/two_pendulums.bioMod"
    ocp, bio_model = prepare_ocp(biorbd_model_path=path, n_shooting=N_SHOOTING)
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))
    print("status", sol.status, "iterations", sol.iterations, "cost", sol.cost)

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    t = np.array(np.concatenate(sol.decision_time()).squeeze())
    n_nodes = states["q_u"].shape[1]
    tau = np.zeros((bio_model.nb_tau, n_nodes))
    tau[[0, 3], :-1] = controls["tau"]  # only the two rotations are actuated
    q, qdot, qddot, lambdas = bio_model.compute_all_states_from_u_iterative(states["q_u"], states["qdot_u"], tau)
    q = np.array(q)

    m = biorbd.Model(path)
    tip, org, resid = [], [], []
    for k in range(q.shape[1]):
        mk = m.markers(q[:, k])
        p1, p3 = mk[1].to_array(), mk[3].to_array()
        tip.append(mk[4].to_array())
        org.append(p1)
        resid.append(p1 - p3)
    org, tip, resid = np.array(org), np.array(tip), np.array(resid)
    print("final time", t[-1], "max |residual| (y,z)", np.abs(resid[:, 1:]).max())
    print("q0", q[:, 0], "qend", q[:, -1])
    np.savez(
        OUT / "holonomic_two_pendulums.npz",
        t=t,
        q=q,
        q_u=np.array(states["q_u"]),
        tau=controls["tau"],
        joint1=org,
        tip=tip,
        residual=resid,
        cost=float(sol.cost),
        iterations=int(sol.iterations),
        converged=int(sol.status == 0),
        n_shooting=N_SHOOTING,
    )


if __name__ == "__main__":
    main()
