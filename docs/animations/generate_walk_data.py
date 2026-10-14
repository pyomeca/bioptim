"""
Simplified hopping with contact phases and an IMPACT transition: REAL bioptim / IPOPT solve to feed ``anim_walk.py``.

Model: models/walk_hopper.bioMod, a planar vertical "leg": light Foot (5 kg, q0 = foot height, carries a rigid contact
along z) + heavy Body (70 kg, q1 = leg length, actuated by an INTERNAL force tau). One hop cycle, three phases:
    phase 0  flight (no contact)              foot starts at rest 0.15 m above the floor and falls, touches at its end
    ---- PhaseTransitionFcn.IMPACT (inelastic, frictionless, point contact) ----
    phase 1  stance (ContactType.RIGID_EXPLICIT), F_z >= 0 (TRACK_EXPLICIT_RIGID_CONTACT_FORCES, min_bound=0), F_z = 0 at take-off
    ---- PhaseTransitionFcn.CONTINUOUS (take-off) ----
    phase 2  flight, ends at the apex (same configuration and rest as the start, so the hop is periodic)
Objective: MINIMIZE_CONTROL on tau in every phase. Output: data/walk_hopper.npz

Usage (env with bioptim on the PATH), from the repo root:
    PYTHONPATH=. python docs/animations/generate_walk_data.py
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
    DynamicsOptionsList,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    PhaseTransitionFcn,
    PhaseTransitionList,
    Solver,
    SolutionMerge,
    TorqueBiorbdModel,
)

MODEL = str(Path(__file__).parent / "models" / "walk_hopper.bioMod")
OUT = Path(__file__).parent / "data"
NS = (10, 20, 12)
TS = (float(np.sqrt(2 * 0.15 / 9.81)), 0.3, 0.2)  # phase 0: exact free fall from 0.15 m
Z0, L0 = 0.15, 0.9  # apex: foot height, leg length (start and end of the cycle)
TAU_MAX = 5000.0
TAU_FLIGHT = 400.0  # the leg is a weak spring in the air


def build(w=1e-4, guess=None, ts=TS, contact_min=True):
    models = (
        TorqueBiorbdModel(MODEL),
        TorqueBiorbdModel(MODEL, contact_types=[ContactType.RIGID_EXPLICIT]),
        TorqueBiorbdModel(MODEL),
    )
    obj, cons = ObjectiveList(), ConstraintList()
    xb, ub, xi, ui = BoundsList(), BoundsList(), InitialGuessList(), InitialGuessList()
    dyn = DynamicsOptionsList()
    for p in range(3):
        obj.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=w, phase=p)
        xb.add("q", bounds=models[p].bounds_from_ranges("q"), phase=p)
        xb.add("qdot", bounds=models[p].bounds_from_ranges("qdot"), phase=p)
        tm = {0: 0.0, 1: TAU_MAX, 2: TAU_FLIGHT}[p]
        ub.add("tau", min_bound=[-tm, -tm], max_bound=[tm, tm], phase=p)
        ub[p]["tau"][0, :] = 0  # nothing acts on the foot except the floor and the leg
        dyn.add(DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=5)))
    xb[0]["q"][:, 0] = [Z0, L0]
    xb[0]["qdot"][:, 0] = 0
    xb[0]["q"][0, -1] = 0  # touch-down: the foot reaches the floor
    xb[1]["q"][0, :] = 0  # stance: foot on the floor
    xb[1]["qdot"][0, :] = 0
    xb[2]["q"][:, -1] = [Z0, L0]  # periodic: back to the apex configuration, at rest
    xb[2]["qdot"][:, -1] = 0
    if contact_min:
        cons.add(
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            node=Node.ALL_SHOOTING,
            contact_index=0,
            min_bound=0,
            max_bound=np.inf,
            phase=1,
        )
        cons.add(
            ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
            node=Node.PENULTIMATE,
            contact_index=0,
            min_bound=0,
            max_bound=0,
            phase=1,
        )
    trans = PhaseTransitionList()
    trans.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=0)
    trans.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=1)
    for p in range(3):
        if guess is None:
            z = [[Z0, 0.0], [0, 0], [0, Z0]][p]
            l = [[L0, L0], [L0, L0 - 0.2], [L0 - 0.2, L0]][p]
            xi.add("q", np.array([z, l]), interpolation=InterpolationType.LINEAR, phase=p)
        else:
            xi.add("q", guess[p][0], interpolation=InterpolationType.EACH_FRAME, phase=p)
            xi.add("qdot", guess[p][1], interpolation=InterpolationType.EACH_FRAME, phase=p)
            ui.add("tau", guess[p][2], interpolation=InterpolationType.EACH_FRAME, phase=p)
    return OptimalControlProgram(
        models,
        NS,
        ts,
        dynamics=dyn,
        x_bounds=xb,
        u_bounds=ub,
        x_init=xi,
        u_init=ui,
        objective_functions=obj,
        constraints=cons,
        phase_transitions=trans,
        use_sx=True,
    )


def solve(ocp, iters=300):
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(int(__import__("os").environ.get("PL", 0)))
    solver.set_maximum_iterations(iters)
    return ocp.solve(solver)


def extract(sol, ocp):
    st = sol.decision_states(to_merge=SolutionMerge.NODES)
    ct = sol.decision_controls(to_merge=SolutionMerge.NODES)
    tm = sol.decision_time(to_merge=SolutionMerge.NODES)
    out = {}
    for p in range(3):
        q, qd = np.array(st[p]["q"]), np.array(st[p]["qdot"])
        tau = np.array(ct[p]["tau"])
        out[f"p{p}_t"] = np.array(tm[p]).ravel()
        out[f"p{p}_q"], out[f"p{p}_qdot"], out[f"p{p}_tau"] = q, qd, tau
    fn = ocp.nlp[1].model.rigid_contact_forces()
    q, qd, tau = out["p1_q"], out["p1_qdot"], out["p1_tau"]
    out["p1_F"] = np.array([float(fn(q[:, k], qd[:, k], tau[:, k], [], [])[0]) for k in range(tau.shape[1])])
    out["cost"], out["status"], out["iterations"] = float(sol.cost), int(sol.status), int(sol.iterations)
    return out


if __name__ == "__main__":
    t0 = time.time()
    ocp = build()
    res = extract(solve(ocp), ocp)
    res.update(ns=np.array(NS), phase_times=np.array(TS), foot_mass=5.0, body_mass=70.0, tau_flight=TAU_FLIGHT)
    # same problem WITHOUT the two force constraints: is F >= 0 active?
    ocp_f = build(contact_min=False, guess=[(res[f"p{p}_q"], res[f"p{p}_qdot"], res[f"p{p}_tau"]) for p in range(3)])
    free = extract(solve(ocp_f), ocp_f)
    res.update(free_F_min=float(free["p1_F"].min()), free_status=free["status"], free_cost=free["cost"])
    print("status", res["status"], "iters", res["iterations"], f"cost {res['cost']:.2f}", f"{time.time() - t0:.0f}s")
    print("no force constraints: status", free["status"], f"cost {free['cost']:.2f}", f"F min {free['p1_F'].min():.0f}")
    print("F (N):", np.round(res["p1_F"], 0))
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "walk_hopper.npz"
    np.savez_compressed(out, **res)
