"""
Direct collocation: effect of ``polynomial_degree`` and of the point family (``method="legendre"`` or ``"radau"``) on
accuracy. Everything stored in ``data/colloc_pendulum.npz`` is REAL bioptim / IPOPT / scipy output (cart-pendulum swing (rotation from 0 to 1 rad, sliding translation actuated),
model of generate_accuracy_data.py, N = 30 intervals, T = 1 s). Only ``OdeSolver.COLLOCATION(...)`` changes.

Two error measures of each optimised solution against the dynamics (reference = DOP853, rtol 1e-10, atol 1e-12, same
dynamics function of the OCP, controls constant on each interval):
    drift_*  : integrate from x0 over the whole horizon, never reset (Shooting.SINGLE like); error on the final state
               and on theta at every node.  Amplified by the swing-up.
    local_*  : integrate each interval from the OPTIMISED node state; error on the next node (Shooting.MULTIPLE like).
The collocation points are those used by the integrator: ``[0] + casadi.collocation_points(d, method)``.

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_colloc_data.py
"""

import time
from pathlib import Path

import numpy as np
from casadi import collocation_points
from scipy.integrate import solve_ivp
from bioptim import (
    BoundsList,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    Objective,
    ObjectiveFcn,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
import os

TARGET = float(os.environ.get("COLLOC_TARGET", 1.0))
ROT, T, N = 1, float(os.environ.get("COLLOC_T", 1.0)), int(os.environ.get("COLLOC_N", 30))
DEGREES = [2, 3, 4, 5, 6]
METHODS = ["legendre", "radau"]


def prepare_ocp(degree, method):
    bio_model = TorqueBiorbdModel(MODEL)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][ROT, -1] = TARGET
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * bio_model.nb_tau, [100] * bio_model.nb_tau
    u_bounds["tau"][ROT, :] = 0  # the rotation is passive, the sliding translation is actuated
    x_init = InitialGuessList()
    x_init.add("q", np.array([[0, 0], [0, TARGET]]), interpolation=InterpolationType.LINEAR)
    return OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.COLLOCATION(polynomial_degree=degree, method=method)),
        x_bounds=x_bounds,
        x_init=x_init,
        u_bounds=u_bounds,
        objective_functions=Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"),
        use_sx=True,
    )


def integrate_interval(nlp, k, x, tau_k):
    dt = T / N
    f = lambda t, y: np.array(nlp.dynamics_func(t, y, tau_k, [], [], [])).ravel()
    return solve_ivp(f, [k * dt, (k + 1) * dt], x, method="DOP853", rtol=1e-10, atol=1e-12).y[:, -1]


def main():
    res = {}
    for method in METHODS:
        for deg in DEGREES:
            tag = f"{method}{deg}"
            ocp = prepare_ocp(deg, method)
            solver = Solver.IPOPT(show_online_optim=False)
            solver.set_print_level(0)
            solver.set_maximum_iterations(1000)
            t0 = time.time()
            sol = ocp.solve(solver)
            dt = time.time() - t0
            st, ct = sol.decision_states(), sol.decision_controls()
            x = np.array([np.concatenate([st["q"][k][:, 0], st["qdot"][k][:, 0]]) for k in range(N + 1)]).T
            tau = np.array([ct["tau"][k][:, 0] for k in range(N)]).T
            nlp = sol.ocp.nlp[0]
            # global drift
            xr = [x[:, 0].copy()]
            for k in range(N):
                xr.append(integrate_interval(nlp, k, xr[-1], tau[:, k]))
            xr = np.array(xr).T
            # local one-step error
            xl = np.array([integrate_interval(nlp, k, x[:, k], tau[:, k]) for k in range(N)]).T
            local = np.linalg.norm(xl - x[:, 1:], axis=0)
            d = dict(
                degree=deg,
                method=method,
                cost=float(sol.cost),
                status=int(sol.status),
                iterations=int(sol.iterations),
                solve_time=dt,
                n_variables=int(np.array(sol.vector).size),
                q_opt=x[:2],
                tau=tau,
                theta_ref=xr[ROT],
                drift_final=float(np.linalg.norm(xr[:, -1] - x[:, -1])),
                drift_theta_max=float(np.abs(xr[ROT] - x[ROT]).max()),
                local_max=float(local.max()),
                local_final=float(local[-1]),
                points=np.array([0.0] + [float(p) for p in collocation_points(deg, method)]),
            )
            res[tag] = d
            print(
                f"{tag}: status={d['status']} it={d['iterations']} nvar={d['n_variables']} t={dt:.2f}s cost={d['cost']:.4f} "
                f"drift_final={d['drift_final']:.2e} drift_theta_max={d['drift_theta_max']:.2e} local_max={d['local_max']:.2e}"
            )
    flat = {f"{tag}_{k}": v for tag, d in res.items() for k, v in d.items()}
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / os.environ.get("COLLOC_OUT", "colloc_pendulum.npz"), **flat)


if __name__ == "__main__":
    main()
