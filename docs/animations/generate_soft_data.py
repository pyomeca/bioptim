"""
Soft (compliant) contact: ContactType.SOFT_EXPLICIT on a ball pushed onto a soft ground. REAL bioptim / IPOPT solves.

Model: models/soft_ball.bioMod (1 kg ball, vertical translation z, one soft-contact sphere of radius 0.05 m at its
centre, ground at z = 0; biorbd SoftContactSphere).  The sphere is penetrating when z < radius, depth = radius - z.
Task: push the ball from Z0 (contact 1 cm above the ground) down to Z1 (3 cm of penetration) in T seconds along a
half-cosine reference (ObjectiveFcn.Lagrange.TRACK_STATE on q, weight 1e3) with a small ObjectiveFcn.Lagrange.
MINIMIZE_CONTROL (1e-6).  The three solves only differ by the ``stiffness`` written in the bioMod
(the file is copied with another number in a temporary folder), warm started one from the other (soft to stiff).
The contact force is the one biorbd computes (``bio_model.soft_contact_forces()``, index 5 = vertical force).

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_soft_data.py
"""

import re
import tempfile
import time
from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
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

HERE = Path(__file__).parent
MODEL = HERE / "models" / "soft_ball.bioMod"
OUT = HERE / "data"
RADIUS = 0.05
Z0, Z1 = 0.06, 0.02
N, T = 50, 0.5
STIFFNESS = [1e4, 1e5, 1e6]
WEIGHT_TRACK = 1e3
WEIGHT_TAU = 1e-6
TAU_MAX = 5000.0
TMP = Path(tempfile.mkdtemp())


def model_path(stiffness):
    text = MODEL.read_text().replace("stiffness 1e4", f"stiffness {stiffness:g}")
    path = TMP / f"soft_ball_k{stiffness:g}.bioMod"
    path.write_text(text)
    return str(path)


def reference(n=N):
    """Height of the ball centre: half-cosine push from Z0 to Z1 (the contact is touched when z = RADIUS)."""
    s = np.linspace(0, 1, n + 1)
    return (Z0 + (Z1 - Z0) * 0.5 * (1 - np.cos(np.pi * s)))[None, :]


def prepare_ocp(stiffness, guess=None, n=N, ode_solver=None):
    bio_model = TorqueBiorbdModel(model_path(stiffness), contact_types=[ContactType.SOFT_EXPLICIT])
    obj = ObjectiveList()
    obj.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=WEIGHT_TAU)
    obj.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=WEIGHT_TRACK, node=Node.ALL, target=reference(n))
    xb = BoundsList()
    xb["q"] = bio_model.bounds_from_ranges("q")
    xb["q"][0, 0] = Z0
    xb["qdot"] = bio_model.bounds_from_ranges("qdot")
    xb["qdot"].min[:, [0]] = 0
    xb["qdot"].max[:, [0]] = 0
    ub = BoundsList()
    ub["tau"] = [-TAU_MAX], [TAU_MAX]
    xi = InitialGuessList()
    ui = InitialGuessList()
    if guess is None:
        xi.add("q", np.array([[Z0, Z1]]), interpolation=InterpolationType.LINEAR)
    else:
        xi.add("q", guess["q"], interpolation=InterpolationType.EACH_FRAME)
        xi.add("qdot", guess["qdot"], interpolation=InterpolationType.EACH_FRAME)
        ui.add("tau", guess["tau"], interpolation=InterpolationType.EACH_FRAME)
    return OptimalControlProgram(
        bio_model,
        n,
        T,
        dynamics=DynamicsOptions(ode_solver=ode_solver or OdeSolver.RK4(n_integration_steps=5)),
        x_bounds=xb,
        u_bounds=ub,
        x_init=xi,
        u_init=ui,
        objective_functions=obj,
        use_sx=False,
    )


def solve(stiffness, guess=None, **kw):
    ocp = prepare_ocp(stiffness, guess, **kw)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(500)
    t0 = time.time()
    sol = ocp.solve(solver)
    dt = time.time() - t0
    st = sol.decision_states(to_merge=SolutionMerge.NODES)
    ct = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qd, tau = st["q"], st["qdot"], ct["tau"]
    f = ocp.nlp[0].model.soft_contact_forces()
    force = np.array([float(np.array(f(q[:, k], qd[:, k], [])).ravel()[5]) for k in range(q.shape[1])])
    res = dict(
        t=np.linspace(0, T, q.shape[1]),
        q=q,
        qdot=qd,
        tau=np.hstack([tau, tau[:, -1:]]) if tau.shape[1] == q.shape[1] - 1 else tau,
        force=force,
        depth=RADIUS - q[0],
        status=sol.status,
        cost=float(sol.cost),
        iterations=int(sol.iterations),
        solve_time=dt,
    )
    return res, sol


def main():
    out = {}
    guess = None
    for k in STIFFNESS:
        res, _ = solve(k, guess)
        print(
            f"k={k:g} status={res['status']} it={res['iterations']} t={res['solve_time']:.1f}s cost={res['cost']:.4g}"
            f" max depth={res['depth'].max():.4f} max F={res['force'].max():.1f} tau max={np.abs(res['tau']).max():.1f}"
        )
        for key, v in res.items():
            out[f"k{k:g}_{key}"] = v
        guess = dict(q=res["q"], qdot=res["qdot"], tau=res["tau"][:, :-1])
    # integration step at the stiffest contact: same problem, 1 / 5 / 20 RK4 sub-steps per interval (warm started)
    kmax = STIFFNESS[-1]
    for ns in (1, 5, 20):
        res, _ = solve(kmax, guess, ode_solver=OdeSolver.RK4(n_integration_steps=ns))
        print(f"RK4 x{ns}: status={res['status']} it={res['iterations']} max F={res['force'].max():.1f}")
        out[f"steps{ns}_force"] = res["force"]
        out[f"steps{ns}_depth"] = res["depth"]
        out[f"steps{ns}_status"] = res["status"]
        out[f"steps{ns}_iterations"] = res["iterations"]
    # the force law itself, as computed by biorbd: F(depth) at rest and while sinking at 0.1 m/s
    depths = np.linspace(0, 0.04, 81)
    out["law_depth"] = depths
    for k in STIFFNESS:
        f = TorqueBiorbdModel(model_path(k), contact_types=[ContactType.SOFT_EXPLICIT]).soft_contact_forces()
        for name, v in (("rest", 0.0), ("sink", -0.1)):
            out[f"law{k:g}_{name}"] = np.array([float(np.array(f([RADIUS - d], [v], [])).ravel()[5]) for d in depths])
        sel = (depths >= 0.005) & (depths <= 0.03)
        exponent = np.polyfit(np.log(depths[sel]), np.log(out[f"law{k:g}_rest"][sel]), 1)[0]
        print(
            f"k={k:g}: log-log slope of F(depth) at rest = {exponent:.3f}; F(3 cm) = {out[f'law{k:g}_rest'][60]:.2f} N"
        )
    out["stiffness"] = np.array(STIFFNESS)
    out["radius"] = RADIUS
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "soft_ball.npz", **out)


if __name__ == "__main__":
    main()
