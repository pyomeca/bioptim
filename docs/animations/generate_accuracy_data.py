"""
Consistency check of a bioptim solution: re-integrate the optimal controls from the initial state with
``sol.integrate(shooting_type=Shooting.SINGLE, integrator=...)`` and compare with the optimised states.
Everything stored in ``data/accuracy_*.npz`` is REAL bioptim / IPOPT / scipy output (pendulum swing-up).

Four OCPs (same problem, T = 1 s, only the transcription changes):
    rk4_coarse   N = 30, OdeSolver.RK4(n_integration_steps=1)
    rk4_fine     N = 30, OdeSolver.RK4(n_integration_steps=5)
    col3         N = 30, OdeSolver.COLLOCATION(polynomial_degree=3)
    col5         N = 30, OdeSolver.COLLOCATION(polynomial_degree=5)

Caveat found while building this: for COLLOCATION the interior points returned by integrate() are not usable (time labels
do not match the values), so only the values at the shooting nodes are used. Reference integrator: SolutionIntegrator.SCIPY_DOP853 (rtol/atol of scipy defaults are loose, so we do not rely on
them for RK4: for RK4 we also store the SolutionIntegrator.OCP re-integration, which uses the very same RK4 as the OCP).
Note: this bioptim version has no Shooting.SINGLE_CONTINUOUS; Shooting.SINGLE is the "never reset the state" mode.

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_accuracy_data.py
"""

import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from bioptim import (
    BoundsList,
    DynamicsOptions,
    Objective,
    ObjectiveFcn,
    OdeSolver,
    OptimalControlProgram,
    Shooting,
    SolutionIntegrator,
    SolutionMerge,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
ROT, T = 1, 1.0

CASES = {
    "rk4_coarse": (30, OdeSolver.RK4(n_integration_steps=1)),
    "rk4_fine": (30, OdeSolver.RK4(n_integration_steps=5)),
    "col3": (30, OdeSolver.COLLOCATION(polynomial_degree=3, method="legendre")),
    "col5": (30, OdeSolver.COLLOCATION(polynomial_degree=5, method="legendre")),
}


def prepare_ocp(n, ode_solver):
    bio_model = TorqueBiorbdModel(MODEL)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][ROT, -1] = 3.14
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * bio_model.nb_tau, [100] * bio_model.nb_tau
    u_bounds["tau"][ROT, :] = 0
    return OptimalControlProgram(
        bio_model,
        n,
        T,
        dynamics=DynamicsOptions(ode_solver=ode_solver),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"),
        use_sx=True,
    )


def dense(sol, integrator):
    """
    Re-integrate from the initial state, never resetting the state (Shooting.SINGLE).
    Returns the states at the shooting nodes (4, N+1) and the points inside the intervals (t, states (4, M)).
    """
    out, t = sol.integrate(shooting_type=Shooting.SINGLE, integrator=integrator, return_time=True)
    x = [np.vstack([out["q"][k], out["qdot"][k]]) for k in range(len(out["q"]))]
    nodes = np.array([xk[:, 0] for xk in x]).T
    t_all = np.concatenate([np.array(tk).ravel() for tk in t])
    return nodes, t_all, np.hstack(x)


def tight_reference(sol, x0, tau):
    """
    Same re-integration (constant tau per interval, from x0, never reset) but with rtol = 1e-10, using the dynamics
    function of the OCP. Needed because sol.integrate calls scipy with its default tolerances (rtol = 1e-3), which is
    NOT accurate enough to serve as ground truth for the collocation solutions.
    """
    nlp = sol.ocp.nlp[0]
    n, dt = nlp.ns, T / nlp.ns
    x = np.array(x0, dtype=float)
    out = [x.copy()]
    for k in range(n):
        f = lambda t, y, k=k: np.array(nlp.dynamics_func(t, y, tau[:, k], [], [], [])).ravel()
        x = solve_ivp(f, [k * dt, (k + 1) * dt], x, method="DOP853", rtol=1e-10, atol=1e-12).y[:, -1]
        out.append(x.copy())
    return np.array(out).T


def main():
    res = {}
    for tag, (n, ode) in CASES.items():
        ocp = prepare_ocp(n, ode)
        solver = Solver.IPOPT(show_online_optim=False)
        solver.set_print_level(0)
        solver.set_maximum_iterations(500)
        t0 = time.time()
        sol = ocp.solve(solver)
        dt = time.time() - t0
        states = sol.decision_states()
        q = np.array([states["q"][k][:, 0] for k in range(n + 1)]).T
        qd = np.array([states["qdot"][k][:, 0] for k in range(n + 1)]).T
        tn = np.linspace(0, T, n + 1)  # fixed final time
        tau = np.array([sol.decision_controls()["tau"][k][:, 0] for k in range(n)]).T
        nvar = int(np.array(sol.vector).size)
        x_nodes, t_i, x_i = dense(sol, SolutionIntegrator.SCIPY_DOP853)
        x_opt = np.vstack([q, qd])
        x_ref = tight_reference(sol, x_opt[:, 0], tau)
        print(
            f"  bioptim DOP853 (rtol 1e-3) vs tight reference at final node: {np.linalg.norm(x_nodes[:, -1] - x_ref[:, -1]):.2e}"
        )
        x_nodes_bioptim = x_nodes
        x_nodes = x_ref  # ground truth for the drift
        err = np.abs(x_nodes[ROT] - x_opt[ROT])  # rotation drift (rad) at every node
        d = {
            "N": n,
            "cost": float(sol.cost),
            "status": int(sol.status),
            "iterations": int(sol.iterations),
            "solve_time": dt,
            "n_variables": nvar,
            "t_nodes": tn,
            "q_opt": q,
            "qdot_opt": qd,
            "tau": tau,
            "x_nodes": x_nodes,
            "x_nodes_bioptim_dop853": x_nodes_bioptim,
            "t_int": t_i,
            "x_int": x_i,
            "err_theta_nodes": err,
            "final_state_error": float(np.linalg.norm(x_nodes[:, -1] - x_opt[:, -1])),
            "final_theta_error": float(err[-1]),
            "max_theta_error": float(err.max()),
        }
        if tag.startswith("rk4"):
            xn_o, t_o, x_o = dense(sol, SolutionIntegrator.OCP)
            d["t_int_ocp"], d["x_int_ocp"] = t_o, x_o
            d["final_state_error_ocp"] = float(np.linalg.norm(xn_o[:, -1] - x_opt[:, -1]))
        res[tag] = d
        print(
            f"{tag}: status={d['status']} iters={d['iterations']} nvar={nvar} time={dt:.2f}s "
            f"final err={d['final_state_error']:.3e} max theta err={d['max_theta_error']:.3e}"
            + (f" (OCP integrator: {d['final_state_error_ocp']:.3e})" if "final_state_error_ocp" in d else "")
        )
    flat = {f"{tag}_{k}": v for tag, d in res.items() for k, v in d.items()}
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "accuracy_pendulum.npz", **flat)


if __name__ == "__main__":
    main()
