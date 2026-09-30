"""
Muscle-driven reaching A -> B (arm26, 2 dof, 6 muscles) versus the torque-driven reach of the same model. REAL IPOPT
solves stored in ``data/muscfull_arm.npz`` for ``anim_muscfull.py``.

Both problems also carry a small smoothing term (Lagrange MINIMIZE_STATE on qdot, weight 0.01).
Both problems: same model (arm26_muscle_driven_ocp.bioMod), N = 30 shooting nodes, T = 0.8 s, RK4, q fixed at A on the
first node and at B on the last node, qdot = 0 at both ends. Muscle problem: MusclesBiorbdModel (no residual torque),
Lagrange MINIMIZE_CONTROL on "muscles" (integral of activation^2, an effort proxy, NOT a metabolic cost). Torque problem:
TorqueBiorbdModel, Lagrange MINIMIZE_CONTROL on "tau".

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_muscfull_data.py
"""

import os
from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    MusclesBiorbdModel,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    SolutionMerge,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/arm26_muscle_driven_ocp.bioMod"
OUT = Path(__file__).parent / "data"
N, T = 30, float(os.environ.get("MF_T", 0.8))
WQ = float(os.environ.get("MF_WQ", 0.01))  # weight of the small smoothing term on qdot (same in both problems)
QA = np.array([0.30, 0.50])  # shoulder flexion, elbow flexion at A [rad]
QB = np.array([0.90, 1.50])  # at B [rad]
HAND = "COM_hand"


def prepare(muscles: bool, weight_key: str):
    model = MusclesBiorbdModel(MODEL, with_residual_torque=False) if muscles else TorqueBiorbdModel(MODEL)
    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = QA
    x_bounds["q"][:, -1] = QB
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0
    x_init = InitialGuessList()
    x_init.add("q", np.array([QA, QB]).T, interpolation=InterpolationType.LINEAR)
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    if muscles:
        u_bounds["muscles"] = [0.0] * model.nb_muscles, [1.0] * model.nb_muscles
        u_init["muscles"] = [0.2] * model.nb_muscles
    else:
        u_bounds["tau"] = [-50.0] * model.nb_tau, [50.0] * model.nb_tau
    obj = ObjectiveList()
    obj.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key=weight_key)
    obj.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=WQ)
    ocp = OptimalControlProgram(
        model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=5)),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=obj,
        n_threads=1,
    )
    return ocp, model


def solve(ocp):
    solver = Solver.IPOPT()
    solver.set_print_level(0)
    solver.set_maximum_iterations(1000)
    solver.set_linear_solver("mumps")
    sol = ocp.solve(solver)
    return sol


def main():
    import biorbd

    bm = biorbd.Model(MODEL)
    names = [bm.markerNames()[i].to_string() for i in range(bm.nbMarkers())]
    ih = names.index(HAND)
    mnames = [bm.muscleNames()[i].to_string() for i in range(bm.nbMuscles())]
    res = {}
    for tag, muscles, key in (("mus", True, "muscles"), ("tor", False, "tau")):
        ocp, model = prepare(muscles, key)
        sol = solve(ocp)
        st = sol.decision_states(to_merge=SolutionMerge.NODES)
        ct = sol.decision_controls(to_merge=SolutionMerge.NODES)
        q, qd = np.array(st["q"]), np.array(st["qdot"])
        hand = np.array([bm.markers(q[:, k])[ih].to_array() for k in range(N + 1)])
        print(tag, "status", sol.status, "iters", sol.iterations, "cost", float(sol.cost))
        res[f"{tag}_q"], res[f"{tag}_qdot"], res[f"{tag}_hand"] = q, qd, hand
        res[f"{tag}_status"], res[f"{tag}_iters"], res[f"{tag}_cost"] = sol.status, sol.iterations, float(sol.cost)
        if muscles:
            act = np.array(ct["muscles"])  # (6, N) on the intervals (last node has no control here)
            f = model.muscle_joint_torque()
            tau = np.array([np.array(f(act[:, k], q[:, k], qd[:, k], [])).ravel() for k in range(act.shape[1])]).T
            res["mus_act"], res["mus_tau"] = act, tau
        else:
            res["tor_tau"] = np.array(ct["tau"])
    res["t"] = np.linspace(0, T, N + 1)
    res["muscle_names"] = np.array(mnames)
    res["QA"], res["QB"] = QA, QB
    print("muscles", mnames)
    print("peak act", res["mus_act"].max(axis=1).round(3))
    print("tau mus peak", np.abs(res["mus_tau"]).max(axis=1), "tau tor peak", np.abs(res["tor_tau"]).max(axis=1))
    print("last-node hand mus", res["mus_hand"][-1], "tor", res["tor_hand"][-1])
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "muscfull_arm.npz", **res)


if __name__ == "__main__":
    main()
