"""
Bounds, initial guess and InterpolationType, with REAL bioptim objects and solves, to feed ``anim_bounds.py``.

Part 1 (real library objects, no solve): the same guess for the pendulum rotation q_rot is given to
``PathCondition`` / ``InitialGuess`` with four ``InterpolationType`` (CONSTANT, CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
LINEAR, EACH_FRAME); the array shape that the user passes and the value used at each node (``evaluate_at``) are stored.
The bounds of q_rot are a real ``Bounds`` object (default interpolation CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT).

Part 2 (real solves): the same pendulum swing-up (pendulum.bioMod, N = 20, T = 1 s, RK4, MINIMIZE_CONTROL on tau) is
solved from three initial guesses: zeros (CONSTANT), a straight line from the start to the end pose (LINEAR) and a
guess taken from a first solve of a COARSE problem (5 intervals) interpolated on the 21 nodes (EACH_FRAME, tag
"fromcoarse"); the coarse solve is a warm start / continuation, its own cost and iterations are stored (``coarse_*``).
IPOPT status and iterations are stored for every solve.

Output: data/bounds_guess.npz

Usage (env with bioptim on the PATH), from the repo root:
    PYTHONPATH=. python docs/animations/generate_bounds_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    DynamicsOptions,
    InitialGuess,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

IT = InterpolationType  # short alias, used in the code panel of the video

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
ROT, N, T, TAU_MAX, N_COARSE, Q_END = 1, 20, 1.0, 100.0, 5, 3.14


def build(n_shooting, x_init, u_init):
    bio_model = TorqueBiorbdModel(MODEL)
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1.0, node=Node.ALL_SHOOTING)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"][:, 0] = 0
    x_bounds["q"][ROT, -1] = Q_END
    x_bounds["qdot"][:, -1] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-TAU_MAX] * bio_model.nb_tau, [TAU_MAX] * bio_model.nb_tau
    u_bounds["tau"][ROT, :] = 0
    return OptimalControlProgram(
        bio_model,
        n_shooting,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=5)),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objectives,
        use_sx=True,
    )


def guess(x, u, interp):
    x_init, u_init = InitialGuessList(), InitialGuessList()
    x_init.add("q", x[:2], interpolation=interp)  # on screen: guess = x[:2]; the same is done for "qdot" and "tau"
    x_init.add("qdot", x[2:], interpolation=interp)
    u_init.add("tau", u, interpolation=interp)
    return x_init, u_init


def solve(ocp):
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(500)
    sol = ocp.solve(solver)
    q = np.array([sol.decision_states()["q"][k][:, 0] for k in range(len(sol.decision_states()["q"]))]).T
    qd = np.array([sol.decision_states()["qdot"][k][:, 0] for k in range(len(sol.decision_states()["qdot"]))]).T
    tau = np.array([sol.decision_controls()["tau"][k][:, 0] for k in range(len(sol.decision_controls()["tau"]))]).T
    return sol, q, qd, tau


if __name__ == "__main__":
    out = {"n_shooting": N, "final_time": T, "q_end": Q_END, "n_coarse": N_COARSE}

    # ---- coarse solve (5 intervals) from zeros -> the "fromcoarse" guess (warm start / continuation) -----------------
    zeros_x, zeros_u = np.zeros((4, 1)), np.zeros((2, 1))
    ocp_c = build(N_COARSE, *guess(zeros_x, zeros_u, IT.CONSTANT))
    sol_c, qc, qdc, tauc = solve(ocp_c)
    out["coarse_status"], out["coarse_iterations"], out["coarse_cost"] = (
        int(sol_c.status),
        int(sol_c.iterations),
        float(sol_c.cost),
    )
    print("coarse", "status", sol_c.status, "iters", sol_c.iterations, "cost", sol_c.cost)
    tc = np.linspace(0, T, N_COARSE + 1)
    tf = np.linspace(0, T, N + 1)
    xc = np.vstack([qc, qdc])
    fc_x = np.array([np.interp(tf, tc, xc[i]) for i in range(4)])  # (4, 21): each row linear between coarse nodes
    fc_u = np.array([tauc[i][np.minimum((np.arange(N) * N_COARSE) // N, N_COARSE - 1)] for i in range(2)])  # (2, 20)
    out["fc_q_rot"] = fc_x[ROT]

    # ---- part 1: the four interpolation types applied to q_rot (real PathCondition / InitialGuess) --------------
    guesses = {
        "constant": (np.array([[0.0]]), IT.CONSTANT),
        "first_last": (np.array([[0.0, Q_END / 2, Q_END]]), IT.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT),
        "linear": (np.array([[0.0, Q_END]]), IT.LINEAR),
        "each_frame": (fc_x[ROT : ROT + 1], IT.EACH_FRAME),
    }
    for tag, (arr, interp) in guesses.items():
        ig = InitialGuess("q_rot", arr, interpolation=interp)
        ig.check_and_adjust_dimensions(1, N)
        out[f"{tag}_shape"] = np.array(ig.init.shape)
        out[f"{tag}_array"] = np.asarray(arr)
        out[f"{tag}_nodes"] = np.array([float(ig.evaluate_at(k)[0]) for k in range(N + 1)])
        print(tag, "shape", arr.shape, "n_shooting attr", ig.init.n_shooting, "nodes", out[f"{tag}_nodes"][:3])

    # real Bounds object of q_rot as in the ocp (default interpolation of Bounds)
    bio_model = TorqueBiorbdModel(MODEL)
    xb = BoundsList()
    xb["q"] = bio_model.bounds_from_ranges("q")
    xb["q"][:, 0] = 0
    xb["q"][ROT, -1] = Q_END
    b = xb["q"]
    b.check_and_adjust_dimensions(2, N)
    out["bound_type"] = str(b.type)
    out["bound_min_shape"], out["bound_max_shape"] = np.array(b.min.shape), np.array(b.max.shape)
    out["bound_min"] = np.array([float(b.min.evaluate_at(k)[ROT]) for k in range(N + 1)])
    out["bound_max"] = np.array([float(b.max.evaluate_at(k)[ROT]) for k in range(N + 1)])
    print("bounds", b.type, b.min.shape, out["bound_min"][:3], out["bound_max"][:3], out["bound_min"][-1])

    # ---- part 2: real solves from three initial guesses ---------------------------------------------------------
    lin_x = np.array([[0.0, 0.0], [0.0, Q_END], [0.0, 0.0], [0.0, 0.0]])  # (4, 2): start -> end pose, zero velocity
    lin_u = np.zeros((2, 2))
    runs = {
        "zeros": (zeros_x, zeros_u, IT.CONSTANT),
        "linear": (lin_x, lin_u, IT.LINEAR),
        "fromcoarse": (fc_x, fc_u, IT.EACH_FRAME),
    }
    for tag, (x, u, interp) in runs.items():
        ocp = build(N, *guess(x, u, interp))
        sol, q, qd, tau = solve(ocp)
        out[f"{tag}_status"], out[f"{tag}_iterations"], out[f"{tag}_cost"] = (
            int(sol.status),
            int(sol.iterations),
            float(sol.cost),
        )
        out[f"{tag}_q_rot"], out[f"{tag}_tau"] = q[ROT], tau[0]
        print(
            tag,
            "status",
            sol.status,
            "iters",
            sol.iterations,
            "cost",
            sol.cost,
            "peak tau",
            np.abs(tau[0]).max(),
            "qrot end",
            q[ROT][-1],
            "max",
            q[ROT].max(),
            "min",
            q[ROT].min(),
        )
    np.savez(OUT / "bounds_guess.npz", **out)
