"""
Free-floating base: a planar body in zero gravity (trunk with a free, NOT actuated root + two arms, see
models/floating_trunk_2arms.bioMod) reorients itself by moving its arms only.  REAL bioptim / IPOPT solve stored in
``data/floating_reorient.npz``.  The angular momentum is computed with ``bio_model.angular_momentum()`` (biorbd, about the
centre of mass) from the optimised nodes; its decomposition L = L_root(qdot_root) + L_joints(qdot_joints) is exact
because L is linear in qdot.

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_floating_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    TorqueFreeFloatingBaseBiorbdModel,
)

HERE = Path(__file__).parent
MODEL = str(HERE / "models" / "floating_trunk_2arms.bioMod")
OUT = HERE / "data"
N, T, TARGET = 30, 2.0, 0.8  # root rotation target (rad)


def prepare_ocp():
    bio_model = TorqueFreeFloatingBaseBiorbdModel(MODEL)
    x_bounds = BoundsList()
    x_bounds["q_roots"] = bio_model.bounds_from_ranges("q_roots")
    x_bounds["q_roots"][:, 0] = 0
    x_bounds["q_roots"][2, -1] = TARGET
    x_bounds["q_joints"] = bio_model.bounds_from_ranges("q_joints")
    x_bounds["q_joints"].min[:, 1:-1] = -1.5  # arms stay within +-1.5 rad
    x_bounds["q_joints"].max[:, 1:-1] = 1.5
    x_bounds["q_joints"][:, [0, -1]] = 0  # the arms start and end in the same shape
    x_bounds["qdot_roots"] = bio_model.bounds_from_ranges("qdot_roots")
    x_bounds["qdot_roots"][:, 0] = 0
    x_bounds["qdot_joints"] = bio_model.bounds_from_ranges("qdot_joints")
    x_bounds["qdot_joints"][:, [0, -1]] = 0
    n_j = bio_model.nb_q - bio_model.nb_root
    u_bounds = BoundsList()
    u_bounds["tau_joints"] = [-100] * n_j, [100] * n_j

    t = np.linspace(0, 1, N + 1)
    x_init = InitialGuessList()
    x_init.add(
        "q_joints",
        1.0 * np.array([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t) - 1]),
        interpolation=InterpolationType.EACH_FRAME,
    )
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau_joints", weight=1e-2)
    return OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=3)),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objectives,
        use_sx=True,
    )


def main():
    ocp = prepare_ocp()
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(2000)
    sol = ocp.solve(solver)
    model = sol.ocp.nlp[0].model
    st, ct = sol.decision_states(), sol.decision_controls()
    qr = np.array([st["q_roots"][k][:, 0] for k in range(N + 1)]).T
    qj = np.array([st["q_joints"][k][:, 0] for k in range(N + 1)]).T
    dr = np.array([st["qdot_roots"][k][:, 0] for k in range(N + 1)]).T
    dj = np.array([st["qdot_joints"][k][:, 0] for k in range(N + 1)]).T
    tau = np.array([ct["tau_joints"][k][:, 0] for k in range(N)]).T
    am = model.angular_momentum()
    q = np.vstack([qr, qj])
    L = np.array([float(am(q[:, k], np.concatenate([dr[:, k], dj[:, k]]), [])[0]) for k in range(N + 1)])
    L_root = np.array([float(am(q[:, k], np.concatenate([dr[:, k], 0 * dj[:, k]]), [])[0]) for k in range(N + 1)])
    L_joints = np.array([float(am(q[:, k], np.concatenate([0 * dr[:, k], dj[:, k]]), [])[0]) for k in range(N + 1)])
    print(f"status={sol.status} it={sol.iterations} cost={float(sol.cost):.4f} root angle end={qr[2, -1]:.4f}")
    print(
        f"max|L|={np.abs(L).max():.2e}  max|L_root|={np.abs(L_root).max():.3f}  max|L_joints|={np.abs(L_joints).max():.3f}"
    )
    print("qdot end", dr[:, -1], dj[:, -1], "|tau| max", np.abs(tau).max())
    OUT.mkdir(exist_ok=True)
    np.savez(
        OUT / "floating_reorient.npz",
        q_roots=qr,
        q_joints=qj,
        qdot_roots=dr,
        qdot_joints=dj,
        tau=tau,
        L=L,
        L_root=L_root,
        L_joints=L_joints,
        T=T,
        N=N,
        status=int(sol.status),
        iterations=int(sol.iterations),
        cost=float(sol.cost),
    )


if __name__ == "__main__":
    main()
