"""
Solve small bioptim OCPs (pendulum swing-up) to feed the Manim scenes of ``features_scenes.py``.
Every number stored in ``data/features_*.npz`` is REAL bioptim / IPOPT output.

Four experiments, one file each:
    features_objectives.npz   Lagrange (MINIMIZE_CONTROL) + Mayer (TRACK_STATE at Node.END) with several Mayer weights
    features_constraints.npz  same swing-up with the torque bound |tau| <= u_max shrinking (+ a bound on the cart position)
    features_multiphase.npz   two phases of different durations, PhaseTransitionFcn.CONTINUOUS vs DISCONTINUOUS
    features_time.npz         ObjectiveFcn.Mayer.MINIMIZE_TIME for several torque bounds (the phase duration is optimized)

Model: bioptim/examples/models/pendulum.bioMod, q = (translation y, rotation theta), tau = (force on y, 0 on theta):
only the sideways force is actuated (tau[1] is bounded to 0), the rotation is passive.

Usage (env with bioptim, biorbd, casadi, IPOPT on the PATH), from the repo root:
    PYTHONPATH=. python docs/animations/generate_features_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    DynamicsOptions,
    DynamicsOptionsList,
    Node,
    Objective,
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
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
ROT = 1  # index of the pendulum rotation in q (index 0 is the sideways translation)
TAU_MAX = 100.0


def as_list(x):
    return x if isinstance(x, list) else [x]


def extract(sol, tag: str) -> dict:
    """Per phase p: t (N+1,), q / qdot (dof, N+1) at the shooting nodes, tau (dof, N). Plus cost, iterations, status."""
    n_phases = len(sol.ocp.nlp)
    states, controls, times = (
        as_list(v)
        for v in (sol.decision_states(), sol.decision_controls(), sol.decision_time(to_merge=SolutionMerge.NODES))
    )
    out = {}
    for p in range(n_phases):
        n = sol.ocp.nlp[p].ns
        out[f"{tag}_p{p}_t"] = np.array(times[p]).ravel()
        out[f"{tag}_p{p}_q"] = np.array([states[p]["q"][k][:, 0] for k in range(n + 1)]).T
        out[f"{tag}_p{p}_qdot"] = np.array([states[p]["qdot"][k][:, 0] for k in range(n + 1)]).T
        out[f"{tag}_p{p}_tau"] = np.array([controls[p]["tau"][k][:, 0] for k in range(n)]).T
    out[f"{tag}_cost"] = float(sol.cost)
    cost = out[f"{tag}_cost"]
    out[f"{tag}_iterations"] = int(sol.iterations)
    out[f"{tag}_converged"] = int(sol.status == 0)
    out[f"{tag}_n_phases"] = n_phases
    print(f"  {tag}: status={sol.status} iters={sol.iterations} cost={cost:.4g} T={out[f'{tag}_p0_t'][-1]:.3f}")
    return out


def solve(ocp):
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(500)
    return ocp.solve(solver)


def pendulum_bounds(bio_model, u_max=TAU_MAX, end_fixed=True, start_fixed=True):
    """Start hanging at rest; end upright (3.14) at rest if ``end_fixed``. Only the translation is actuated."""
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    if start_fixed:
        x_bounds["q"][:, 0] = 0
        x_bounds["qdot"][:, 0] = 0
    if end_fixed:
        x_bounds["q"][ROT, -1] = 3.14
        x_bounds["qdot"][:, -1] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-u_max] * bio_model.nb_tau, [u_max] * bio_model.nb_tau
    u_bounds["tau"][ROT, :] = 0
    return x_bounds, u_bounds


def rk4():
    return DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=5))


# --------------------------------------------------------------------------------------------------------------------
# 1. Objectives: Lagrange vs Mayer
# --------------------------------------------------------------------------------------------------------------------
N1, T1 = 30, 1.0
MAYER_WEIGHTS = [0.0, 1.0, 30.0, 300.0, 10000.0]
LAGRANGE_WEIGHT = 1.0


def ocp_objectives(w_mayer: float):
    bio_model = TorqueBiorbdModel(MODEL)
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=LAGRANGE_WEIGHT, node=Node.ALL_SHOOTING)
    if w_mayer > 0:
        objectives.add(ObjectiveFcn.Mayer.TRACK_STATE, key="q", index=[ROT], target=3.14, weight=w_mayer, node=Node.END)
    x_bounds, u_bounds = pendulum_bounds(bio_model, end_fixed=False)
    return OptimalControlProgram(
        bio_model,
        N1,
        T1,
        dynamics=rk4(),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objectives,
        use_sx=True,
    )


# --------------------------------------------------------------------------------------------------------------------
# 2. Constraints and bounds
# --------------------------------------------------------------------------------------------------------------------
U_MAXS = [100.0, 20.0, 15.0, 12.0]
CART_LIMIT = 0.4


def ocp_constraints(u_max: float, cart_limit: float = None):
    bio_model = TorqueBiorbdModel(MODEL)
    objectives = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    x_bounds, u_bounds = pendulum_bounds(bio_model, u_max=u_max)
    if cart_limit is not None:
        x_bounds["q"].min[0, 1:] = -cart_limit
        x_bounds["q"].max[0, 1:] = cart_limit
    return OptimalControlProgram(
        bio_model,
        N1,
        T1,
        dynamics=rk4(),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objectives,
        use_sx=True,
    )


# --------------------------------------------------------------------------------------------------------------------
# 3. Multiphase
# --------------------------------------------------------------------------------------------------------------------
N_PH = (12, 18)
T_PH = (0.5, 1.0)
MID_ANGLE = 1.57


def ocp_multiphase(transition: str):
    models = (TorqueBiorbdModel(MODEL), TorqueBiorbdModel(MODEL))
    objectives = ObjectiveList()
    for p in range(2):
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=p)
    x_bounds, u_bounds = BoundsList(), BoundsList()
    for p in range(2):
        xb, ub = pendulum_bounds(models[p], end_fixed=False, start_fixed=(p == 0))
        x_bounds.add("q", bounds=xb["q"], phase=p)
        x_bounds.add("qdot", bounds=xb["qdot"], phase=p)
        u_bounds.add("tau", bounds=ub["tau"], phase=p)
    # phase 0 ends at 90 degrees (velocity free); phase 1 is free at its start and ends upright at rest
    x_bounds[0]["q"][ROT, -1] = MID_ANGLE
    x_bounds[1]["q"][ROT, -1] = 3.14
    x_bounds[1]["qdot"][:, -1] = 0
    if transition == "discontinuous":
        # phase 1 restarts at rest from 90 degrees: only possible because the transition does not link the states
        x_bounds[1]["q"][ROT, 0] = MID_ANGLE
        x_bounds[1]["qdot"][:, 0] = 0
    transitions = PhaseTransitionList()
    fcn = PhaseTransitionFcn.CONTINUOUS if transition == "continuous" else PhaseTransitionFcn.DISCONTINUOUS
    transitions.add(fcn, phase_pre_idx=0)
    dynamics = DynamicsOptionsList()
    dynamics.add(rk4())
    dynamics.add(rk4())
    return OptimalControlProgram(
        models,
        N_PH,
        T_PH,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objectives,
        phase_transitions=transitions,
        use_sx=True,
    )


# --------------------------------------------------------------------------------------------------------------------
# 4. Free time
# --------------------------------------------------------------------------------------------------------------------
N4 = 40
T4_GUESS = 1.0
U_MAXS_TIME = [100.0, 80.0, 60.0]


def ocp_time(u_max: float):
    bio_model = TorqueBiorbdModel(MODEL)
    objectives = Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.1, max_bound=4.0)
    x_bounds, u_bounds = pendulum_bounds(bio_model, u_max=u_max)
    return OptimalControlProgram(
        bio_model,
        N4,
        T4_GUESS,
        dynamics=rk4(),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objectives,
        use_sx=True,
    )


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)

    print("1. objectives")
    res = {"mayer_weights": np.array(MAYER_WEIGHTS), "n_shooting": N1, "final_time": T1}
    for i, w in enumerate(MAYER_WEIGHTS):
        res.update(extract(solve(ocp_objectives(w)), f"w{i}"))
    np.savez_compressed(OUT / "features_objectives.npz", **res)

    print("2. constraints")
    res = {"u_maxs": np.array(U_MAXS), "n_shooting": N1, "final_time": T1, "cart_limit": CART_LIMIT}
    for i, u in enumerate(U_MAXS):
        res.update(extract(solve(ocp_constraints(u)), f"u{i}"))
    res.update(extract(solve(ocp_constraints(100.0, cart_limit=CART_LIMIT)), "cart"))
    np.savez_compressed(OUT / "features_constraints.npz", **res)

    print("3. multiphase")
    res = {"n_shooting": np.array(N_PH), "phase_times": np.array(T_PH)}
    for tag in ("continuous", "discontinuous"):
        res.update(extract(solve(ocp_multiphase(tag)), tag))
    np.savez_compressed(OUT / "features_multiphase.npz", **res)

    print("4. free time")
    res = {"u_maxs": np.array(U_MAXS_TIME), "n_shooting": N4, "t_guess": T4_GUESS}
    for i, u in enumerate(U_MAXS_TIME):
        res.update(extract(solve(ocp_time(u)), f"u{i}"))
    np.savez_compressed(OUT / "features_time.npz", **res)
    print("done")
