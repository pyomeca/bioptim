"""
Muscle excitation -> activation dynamics with REAL bioptim / IPOPT output for ``anim_excitation.py``.

Same reaching task as bioptim/examples/toy_examples/muscle_driven_ocp/static_arm.py (arm26, 2 dof, 6 muscles), but the
model is ``MusclesWithExcitationsBiorbdModel``: the excitation e(t) is the CONTROL ("muscles"), the activation a(t)
is a STATE ("muscles"), and biorbd's ``activationDot`` gives da/dt (De Groote 2016 formula, tau_act = 0.01 s,
tau_deact = 0.04 s, b = 0.1). Also stores a dense activation curve obtained by integrating that same casadi function
with the excitation held constant on each interval, and the joint-space / marker trajectories.
Output: data/excitation_arm.npz

Usage (env with bioptim on the PATH), from the repo root:
    PYTHONPATH=. python docs/animations/generate_excitation_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    DynamicsOptions,
    InitialGuessList,
    MusclesWithExcitationsBiorbdModel,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    SolutionMerge,
    Solver,
)
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/arm26_muscle_driven_ocp.bioMod"
OUT = Path(__file__).parent / "data"
N, T, WEIGHT = 30, 0.5, 1000
SUB = 40  # dense sub-steps per interval


def build():
    bio_model = MusclesWithExcitationsBiorbdModel(MODEL, with_residual_torque=True)
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles")
    objectives.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, first_marker="target", second_marker="COM_hand", weight=WEIGHT
    )
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = (0.07, 1.4)
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = 0
    x_bounds["muscles"] = [0.0] * bio_model.nb_muscles, [1.0] * bio_model.nb_muscles  # activation a(t) in [0, 1]
    x_bounds["muscles"][:, 0] = 0.1  # resting activation at t = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-1.0] * bio_model.nb_tau, [1.0] * bio_model.nb_tau
    u_bounds["muscles"] = [0.0] * bio_model.nb_muscles, [1.0] * bio_model.nb_muscles  # excitation e(t) in [0, 1]
    x_init = InitialGuessList()
    x_init["q"] = [1.57] * bio_model.nb_q
    x_init["muscles"] = [0.3] * bio_model.nb_muscles
    u_init = InitialGuessList()
    u_init["muscles"] = [0.5] * bio_model.nb_muscles
    ocp = OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4()),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objectives,
        n_threads=1,
    )
    return ocp, bio_model


def main():
    ocp, bio_model = build()
    solver = Solver.IPOPT()
    solver.set_print_level(0)
    solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver)
    st = sol.decision_states(to_merge=SolutionMerge.NODES)
    ct = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, act = np.array(st["q"]), np.array(st["muscles"])
    exc, tau = np.array(ct["muscles"]), np.array(ct["tau"])
    print(
        "status", sol.status, "iters", sol.iterations, "cost", float(sol.cost), "shapes", q.shape, act.shape, exc.shape
    )

    # dense activation from the same biorbd function, RK4, excitation constant on each interval
    f = bio_model.muscle_activation_dot()
    dt = T / N / SUB
    a = act[:, 0].copy()
    t_d, a_d = [0.0], [a.copy()]
    for k in range(N):
        e = exc[:, k]
        for s in range(SUB):
            g = lambda x: np.array(f(e, x, [])).ravel()
            k1 = g(a)
            k2 = g(a + dt / 2 * k1)
            k3 = g(a + dt / 2 * k2)
            k4 = g(a + dt * k3)
            a = a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t_d.append(k * T / N + (s + 1) * dt)
            a_d.append(a.copy())
    a_d = np.array(a_d).T
    print("max |dense(node) - solution activation|:", np.abs(a_d[:, ::SUB] - act).max())

    names = [bio_model.model.muscleNames()[i].to_string() for i in range(bio_model.nb_muscles)]
    mk = [bio_model.model.markerNames()[i].to_string() for i in range(bio_model.model.nbMarkers())]
    mf = bio_model.markers()
    pos = {
        n: np.array([np.array(mf(q[:, k], []))[:, mk.index(n)] for k in range(N + 1)])
        for n in ("r_acromion", "r_humerus_epicondyle", "COM_hand", "target")
    }
    err = float(np.linalg.norm(pos["target"][-1] - pos["COM_hand"][-1]))
    print("names", names, "marker error", err)
    np.savez(
        OUT / "excitation_arm.npz",
        t=np.linspace(0, T, N + 1),
        q=q,
        act=act,
        exc=exc,
        tau=tau,
        t_dense=np.array(t_d),
        act_dense=a_d,
        muscle_names=np.array(names),
        shoulder=pos["r_acromion"],
        elbow=pos["r_humerus_epicondyle"],
        hand=pos["COM_hand"],
        target=pos["target"],
        marker_error=err,
        cost=float(sol.cost),
        iterations=int(sol.iterations),
        status=int(sol.status),
        n_shooting=N,
        final_time=T,
    )


if __name__ == "__main__":
    main()
