"""
Rigid contact + unilateral contact force (REAL bioptim / IPOPT solves) to feed ``anim_contact.py``.

Model: bioptim/examples/models/3segments_4dof_1contact.bioMod (planar leg, one rigid contact at the foot along y and z),

velocity and height at the end (a jump), with ContactType.RIGID_EXPLICIT dynamics. Two solves:
    free      : nothing constrains the contact force
    unilateral: ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES on the vertical force, min_bound=0
Output: data/contact_leg.npz

Usage (env with bioptim on the PATH), from the repo root:
    PYTHONPATH=. python docs/animations/generate_contact_data.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    ConstraintFcn,
    ConstraintList,
    ContactType,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    SolutionMerge,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

MODEL = str(Path(__file__).parent / "models" / "contact_leg.bioMod")
OUT = Path(__file__).parent / "data"
N, T = 40, 0.4
LEG0, LEG1 = 0.4, 0.9  # leg length at start / end (rest to rest)
TAU_MAX = 5000.0


def build(unilateral, ode=None, n=N, t_final=T, w=1e-4, guess=None):
    bio_model = TorqueBiorbdModel(MODEL, contact_types=[ContactType.RIGID_EXPLICIT])
    obj = ObjectiveList()
    obj.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=w)
    cons = ConstraintList()
    if unilateral:
        cons.add(
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            node=Node.ALL_SHOOTING,
            contact_index=0,
            min_bound=0,
            max_bound=np.inf,
        )
    xb = BoundsList()
    xb["q"] = bio_model.bounds_from_ranges("q")
    xb["qdot"] = bio_model.bounds_from_ranges("qdot")
    xb["q"][:, 0] = [0, LEG0]
    xb["qdot"][:, 0] = 0
    xb["q"][:, -1] = [0, LEG1]
    xb["qdot"][:, -1] = 0
    xb["q"][0, :] = 0  # the foot stays on the floor
    ub = BoundsList()
    ub["tau"] = [-TAU_MAX, -TAU_MAX], [TAU_MAX, TAU_MAX]
    ub["tau"][0, :] = 0  # nothing acts on the foot except the floor and the leg
    xi = InitialGuessList()
    ui = InitialGuessList()
    if guess is None:
        xi.add("q", np.array([[0, 0], [LEG0, LEG1]]), interpolation=InterpolationType.LINEAR)
    else:  # warm start (continuation): guess = (q, qdot, tau) of a previous solution
        xi.add("q", guess[0], interpolation=InterpolationType.EACH_FRAME)
        xi.add("qdot", guess[1], interpolation=InterpolationType.EACH_FRAME)
        ui.add("tau", guess[2], interpolation=InterpolationType.EACH_FRAME)
    return OptimalControlProgram(
        bio_model,
        n,
        t_final,
        dynamics=DynamicsOptions(ode_solver=ode or OdeSolver.RK4(n_integration_steps=5)),
        x_bounds=xb,
        u_bounds=ub,
        x_init=xi,
        u_init=ui,
        objective_functions=obj,
        constraints=cons,
        use_sx=True,
    )


def extract(sol, ocp, tag):
    nlp = ocp.nlp[0]
    st = sol.decision_states(to_merge=SolutionMerge.NODES)
    q, qdot = np.array(st["q"]), np.array(st["qdot"])
    tau = np.array(sol.decision_controls(to_merge=SolutionMerge.NODES)["tau"])  # (2, N) constant per interval
    fn = nlp.model.rigid_contact_forces()  # inputs: q, qdot, tau, external_forces, parameters
    # normal force at the N shooting nodes 0..N-1 (the ones constrained by TRACK_CONTACT_FORCES)
    F = np.array([float(fn(q[:, k], qdot[:, k], tau[:, k], [], [])[0]) for k in range(tau.shape[1])])
    t = np.array(sol.decision_time(to_merge=SolutionMerge.NODES)).ravel()
    return {
        f"{tag}_t": t,
        f"{tag}_q": q,
        f"{tag}_qdot": qdot,
        f"{tag}_tau": tau,
        f"{tag}_F": F,
        f"{tag}_cost": float(sol.cost),
        f"{tag}_status": int(sol.status),
        f"{tag}_iterations": int(sol.iterations),
    }


def solve(unilateral, **kw):
    ocp = build(unilateral, **kw)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(500)
    return ocp, ocp.solve(solver)


if __name__ == "__main__":
    res = {"n_shooting": N, "final_time": T, "leg0": LEG0, "leg1": LEG1, "body_mass": 70.0, "foot_mass": 5.0}
    t0 = time.time()
    ocp, sol = solve(False)
    res.update(extract(sol, ocp, "free"))
    # a cold start of the unilateral problem ends in local infeasibility (IPOPT status 1): warm start from the free solution
    ocp_c, sol_c = solve(True)
    res["cold_unilateral_status"] = int(sol_c.status)
    ocp, sol = solve(True, guess=(res["free_q"], res["free_qdot"], res["free_tau"]))
    res.update(extract(sol, ocp, "unilateral"))
    for tag in ("free", "unilateral"):
        F = res[tag + "_F"]
        print(
            f"{tag}: status {res[tag + '_status']} iter {res[tag + '_iterations']} cost {res[tag + '_cost']:.4g}  "
            f"F min {F.min():.1f} max {F.max():.1f} N"
        )
    print("cold-start unilateral status:", res["cold_unilateral_status"], f"({time.time() - t0:.1f}s)")
    np.savez_compressed(OUT / "contact_leg.npz", **res)
