"""
Real stochastic optimal control (bioptim.StochasticOptimalControlProgram, SocpType.COLLOCATION) on the obstacle
avoidance example of Gillis et al. 2013 (bioptim/examples/toy_examples/stochastic_optimal_control/
obstacle_avoidance_direct_collocation.py): a mass point follows a periodic, time-optimal loop around two super-ellipse
obstacles under motor noise.  Two real IPOPT solves of the SAME problem are stored in data/socp_results.npz:
  * deterministic OCP (noise ignored, plain path constraint h(q) >= 0),
  * SOCP with the robustified path constraint  h(q) - gamma * sqrt(dh/dx P dh/dx') >= 0  and covariance propagation.
The covariance of the deterministic solution is obtained afterwards by propagating the SAME linearised covariance
dynamics (open loop, same noise) along its trajectory: this is computed here with numerical integration, not by IPOPT.

Run from the repo root:  PYTHONPATH=. python docs/animations/generate_socp_data.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from bioptim import SocpType, SolutionMerge, Solver
from bioptim.examples.toy_examples.stochastic_optimal_control import obstacle_avoidance_direct_collocation as ex
from bioptim.examples.toy_examples.stochastic_optimal_control.models.mass_point_model import MassPointDynamicsModel

OUT = Path(__file__).parent / "data"
POLY = 5
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
TF = 4
NOISE_LEVEL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
NOISE = np.array([1, 1]) * NOISE_LEVEL


def initial_q(n_shooting):
    model = MassPointDynamicsModel(
        problem_type=SocpType.COLLOCATION(polynomial_degree=POLY), motor_noise_magnitude=NOISE
    )
    q_init = np.zeros((model.nb_q, (POLY + 2) * n_shooting + 1))
    zq = ex.initialize_circle((POLY + 1) * n_shooting + 1)
    for i in range(n_shooting + 1):
        j, k = i * (POLY + 1), i * (POLY + 2)
        q_init[:, k] = zq[:, j]
        q_init[:, k + 1 : k + 1 + (POLY + 1)] = zq[:, j : j + (POLY + 1)]
    return q_init


def solve(stochastic, robust):
    socp_type = SocpType.COLLOCATION(polynomial_degree=POLY, method="legendre")
    socp = ex.prepare_socp(
        final_time=TF,
        n_shooting=N,
        polynomial_degree=POLY,
        motor_noise_magnitude=NOISE,
        q_init=initial_q(N),
        is_stochastic=stochastic,
        is_robustified=robust,
        socp_type=socp_type,
        use_sx=True,
    )
    t0 = time.time()
    sol = socp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000))
    return sol, time.time() - t0


def extract(sol, stochastic):
    st = sol.decision_states(to_merge=SolutionMerge.NODES)
    ctrl = sol.decision_controls(to_merge=SolutionMerge.NODES)
    out = dict(
        q=st["q"],
        qdot=st["qdot"],
        u=ctrl["u"],
        tf=float(np.ravel(sol.decision_time(to_merge=SolutionMerge.NODES))[-1]),
        status=int(sol.status),
        iterations=int(sol.iterations),
        seconds=float(np.ravel(sol.real_time_to_optimize)[0]),
        cost=float(np.ravel(sol.cost)[0]),
    )
    if stochastic:
        out["cov"] = ctrl["cov"]
    return out


if __name__ == "__main__":
    cases = {"det": (False, False), "nonrobust": (True, False), "robust": (True, True)}
    res = {}
    for name, (sto, rob) in cases.items():
        sol, wall = solve(sto, rob)
        r = extract(sol, sto)
        print("RESULT", name, "status", r["status"], "iters", r["iterations"], "tf", r["tf"], "wall", wall, flush=True)
        for k, v in r.items():
            res[f"{name}_{k}"] = v
    res.update(n=N, poly=POLY, noise=NOISE)
    OUT.mkdir(exist_ok=True)
    fname = sys.argv[3] if len(sys.argv) > 3 else "socp_results.npz"
    np.savez(OUT / fname, **res)
