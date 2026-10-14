"""
Real IPOPT data for ``anim_first.py`` (scene FirstOCP): "My first optimal control problem, step by step".

The pendulum swing-up of ``bioptim/examples/getting_started/basic_ocp.py`` / README "A first practical example",
built with the same lines as the video shows (``prepare_ocp`` below, N = 30 intervals, T = 1 s, RK4 multiple shooting,
IPOPT default options, a single cold start, no continuation): model ``pendulum.bioMod`` (sliding translation q0 +
rotation q1, only the translation is actuated), minimise the squared generalised force, q(0) = 0, q1(T) = 3.14 rad,
zero velocity at both ends, |tau| <= 100, initial guess all zeros.

Stored in ``data/first_pendulum.npz``: the solution (t, q, qdot, tau), the bounds really given to the OCP (read back
from the BoundsList objects), the initial guess, the real biorbd marker positions (y, z) of the model at rest, at the
target pose and along the solution, the cost and the IPOPT status / iteration count.

Usage (from the repo root):  PYTHONPATH=. python docs/animations/generate_first_data.py
"""

from pathlib import Path

import biorbd
import numpy as np
from bioptim import (
    BoundsList,
    DynamicsOptions,
    InitialGuessList,
    Objective,
    ObjectiveFcn,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    SolutionMerge,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
N, T = 30, 1.0  # the OCP call below spells them out (30 intervals, 1 s), as on screen


def prepare_ocp():
    # ---- the code of the video starts here (the model path is shortened to "pendulum.bioMod" on screen)
    bio_model = TorqueBiorbdModel(MODEL)
    dynamics = DynamicsOptions(ode_solver=OdeSolver.RK4())
    objective = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 3.14
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100, -100], [100, 100]
    u_bounds["tau"][1, :] = 0

    x_init = InitialGuessList()
    x_init["q"] = [0, 0]
    x_init["qdot"] = [0, 0]
    u_init = InitialGuessList()
    u_init["tau"] = [0, 0]

    ocp = OptimalControlProgram(
        bio_model,
        30,
        1,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective,
        use_sx=True,
    )
    # ---- end of the code of the video
    return ocp, x_bounds, u_bounds, x_init, u_init


def markers_of(q):
    """(2, n_markers, n_nodes): global (y, z) of every marker of the biorbd model for every column of q."""
    m = biorbd.Model(MODEL)
    out = np.zeros((2, m.nbMarkers(), q.shape[1]))
    for k in range(q.shape[1]):
        for i, mk in enumerate(m.markers(q[:, k])):
            out[:, i, k] = mk.to_array()[1:]
    return out


def main():
    ocp, x_bounds, u_bounds, x_init, u_init = prepare_ocp()
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    sol = ocp.solve(solver)
    status, iterations, cost = int(sol.status), int(sol.iterations), float(sol.cost)
    print("status", status, "iterations", iterations, "cost", cost)
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = np.asarray(states["q"]), np.asarray(states["qdot"]), np.asarray(controls["tau"])
    t = np.linspace(0, T, N + 1)
    print("q", q.shape, "tau", tau.shape, "max|tau0|", float(np.abs(tau[0]).max()), "q1(T)", float(q[1, -1]))
    xb_q, xb_qd, ub = x_bounds["q"], x_bounds["qdot"], u_bounds["tau"]
    out = dict(
        status=status,
        converged=status == 0,
        iterations=iterations,
        cost=cost,
        n_shooting=N,
        final_time=T,
        t=t,
        q=q,
        qdot=qdot,
        tau=tau,
        q_min=np.asarray(xb_q.min),
        q_max=np.asarray(xb_q.max),
        qdot_min=np.asarray(xb_qd.min),
        qdot_max=np.asarray(xb_qd.max),
        tau_min=np.asarray(ub.min),
        tau_max=np.asarray(ub.max),
        x_init_q=np.asarray(x_init["q"].init),
        u_init_tau=np.asarray(u_init["tau"].init),
        markers_rest=markers_of(np.zeros((2, 1))),
        markers_target=markers_of(np.array([[0.0], [3.14]])),
        markers_sol=markers_of(q),
    )
    for k in ("q_min", "q_max", "qdot_min", "tau_min", "tau_max", "x_init_q", "u_init_tau"):
        print(
            k,
            out[k].shape,
            np.round(out[k][:, [0, 1, -1]] if out[k].ndim == 2 and out[k].shape[1] > 2 else out[k], 3).tolist(),
        )
    print("marker_2 rest", out["markers_rest"][:, :, 0].tolist(), "target", out["markers_target"][:, :, 0].tolist())
    np.savez(OUT / "first_pendulum.npz", **out)


if __name__ == "__main__":
    main()
