"""
Solve small bioptim OCPs (pendulum swing-up) to feed the Manim scenes of ``features_scenes.py``.
Every number stored in ``data/features_*.npz`` is REAL bioptim / IPOPT output.

Six experiments, one file each:
    features_objectives.npz   Lagrange (MINIMIZE_CONTROL) + Mayer (TRACK_STATE at Node.END) with several Mayer weights
    features_constraints.npz  same swing-up with the torque bound |tau| <= u_max shrinking (+ bounds |y| <= L on the cart position, continuation)
    features_multiphase.npz   two phases of different durations, PhaseTransitionFcn.CONTINUOUS vs DISCONTINUOUS
    features_time.npz         ObjectiveFcn.Mayer.MINIMIZE_TIME for several torque bounds (the phase duration is optimized)
    features_parameters.npz   a ParameterList entry "max_tau" (peak torque, one value for all nodes) optimized with
                              the trajectory, for several weights of ParameterObjectiveList (MINIMIZE_PARAMETER)
    features_impact.npz       point mass falling on the floor (models/point_floor.bioMod): PhaseTransitionFcn.IMPACT
                              (and CONTINUOUS, which is infeasible there)

Model (all but the impact experiment): bioptim/examples/models/pendulum.bioMod, q = (translation y, rotation theta), tau = (force on y, 0 on theta):
only the sideways force is actuated (tau[1] is bounded to 0), the rotation is passive.

Usage (env with bioptim, biorbd, casadi, IPOPT on the PATH), from the repo root:
    PYTHONPATH=. python docs/animations/generate_features_data.py
"""

import sys
from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    ConstraintList,
    ContactType,
    InitialGuessList,
    InterpolationType,
    ParameterList,
    ParameterObjectiveList,
    VariableScaling,
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
MODEL_IMPACT = str(Path(__file__).parent / "models" / "point_floor.bioMod")
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
    try:
        out[f"{tag}_exit"] = str(sol.ocp.ocp_solver.shaked_ocp_solver.stats()["return_status"])
    except Exception:  # keep the data generation robust to internal changes
        out[f"{tag}_exit"] = ""
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
CART_LIMITS = [
    0.9,
    0.7,
    0.5,
]  # successive bounds |y| <= L (continuation, each solve is warm started by the previous one)
CART_LIMIT_TOO_TIGHT = 0.4  # tried as well: IPOPT does not converge (see FEATURES.md)


def ocp_constraints(u_max: float, cart_limit: float = None, previous=None):
    """``previous``: solution used as initial guess (continuation on the cart bound, see FEATURES.md)."""
    bio_model = TorqueBiorbdModel(MODEL)
    objectives = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    x_bounds, u_bounds = pendulum_bounds(bio_model, u_max=u_max)
    if cart_limit is not None:
        x_bounds["q"].min[0, 1:] = -cart_limit
        x_bounds["q"].max[0, 1:] = cart_limit
    x_init, u_init = InitialGuessList(), InitialGuessList()
    if previous is not None:
        st, co = previous.decision_states(), previous.decision_controls()
        each = InterpolationType.EACH_FRAME
        x_init.add("q", np.array([st["q"][k][:, 0] for k in range(N1 + 1)]).T, interpolation=each)
        x_init.add("qdot", np.array([st["qdot"][k][:, 0] for k in range(N1 + 1)]).T, interpolation=each)
        u_init.add("tau", np.array([co["tau"][k][:, 0] for k in range(N1)]).T, interpolation=each)
    return OptimalControlProgram(
        bio_model,
        N1,
        T1,
        dynamics=rk4(),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
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


# --------------------------------------------------------------------------------------------------------------------
# 5. Parameters: the peak torque "max_tau" is a decision variable shared by all the nodes
# --------------------------------------------------------------------------------------------------------------------
PAR_WEIGHTS = [0.001, 0.01, 0.03, 0.1]
P_MAX = 100.0
P_INIT = 50.0


def max_tau_upper(controller):
    """max_tau - tau >= 0, at every shooting node (the parameter is the same everywhere)."""
    return controller.parameters["max_tau"].cx - controller.controls["tau"].cx[0]


def max_tau_lower(controller):
    """max_tau + tau >= 0."""
    return controller.parameters["max_tau"].cx + controller.controls["tau"].cx[0]


def no_model_change(bio_model, parameter):
    """The parameter does not modify the model: it only appears in constraints and objectives."""
    return


def ocp_parameters(weight: float, previous=None):
    """``previous``: the solution used as initial guess (continuation on the weight, see FEATURES.md)."""
    parameters = ParameterList(use_sx=True)
    parameters.add("max_tau", no_model_change, size=1, scaling=VariableScaling("max_tau", [1]))

    parameter_bounds = BoundsList()
    parameter_bounds.add("max_tau", min_bound=0, max_bound=P_MAX, interpolation=InterpolationType.CONSTANT)
    parameter_init = InitialGuessList()
    parameter_init["max_tau"] = P_INIT
    parameter_objectives = ParameterObjectiveList()
    parameter_objectives.add(ObjectiveFcn.Parameter.MINIMIZE_PARAMETER, key="max_tau", weight=weight, quadratic=True)

    bio_model = TorqueBiorbdModel(MODEL, parameters=parameters)
    objectives = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    constraints = ConstraintList()
    constraints.add(max_tau_upper, node=Node.ALL_SHOOTING, min_bound=0, max_bound=np.inf)
    constraints.add(max_tau_lower, node=Node.ALL_SHOOTING, min_bound=0, max_bound=np.inf)
    x_bounds, u_bounds = pendulum_bounds(bio_model, u_max=P_MAX)

    x_init, u_init = InitialGuessList(), InitialGuessList()
    if previous is not None:
        st, co = previous.decision_states(), previous.decision_controls()
        each = InterpolationType.EACH_FRAME
        x_init.add("q", np.array([st["q"][k][:, 0] for k in range(N1 + 1)]).T, interpolation=each)
        x_init.add("qdot", np.array([st["qdot"][k][:, 0] for k in range(N1 + 1)]).T, interpolation=each)
        u_init.add("tau", np.array([co["tau"][k][:, 0] for k in range(N1)]).T, interpolation=each)
        parameter_init["max_tau"] = float(previous.parameters["max_tau"][0])
    return OptimalControlProgram(
        bio_model,
        N1,
        T1,
        dynamics=rk4(),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objectives,
        constraints=constraints,
        parameters=parameters,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        parameter_objectives=parameter_objectives,
        use_sx=True,
    )


# --------------------------------------------------------------------------------------------------------------------
# 6. Impact: a point mass falls on the floor (phase 0: flight, phase 1: on the floor)
# --------------------------------------------------------------------------------------------------------------------
N_IMP = (20, 20)
Z0 = 1.0
G = 9.81
T_FALL = float(np.sqrt(2 * Z0 / G))  # duration of the free fall from Z0 to the floor
T_SLIDE = 1.0
X_END = 3.0


def ocp_impact(transition: str):
    """Phase 0: flight from z = Z0. Phase 1: rigid contact with the floor, slide to x = X_END and stop (tau_z = 0)."""
    models = (
        TorqueBiorbdModel(MODEL_IMPACT),
        TorqueBiorbdModel(MODEL_IMPACT, contact_types=[ContactType.RIGID_EXPLICIT]),
    )
    objectives = ObjectiveList()
    x_bounds, u_bounds = BoundsList(), BoundsList()
    for p in range(2):
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=p)
        x_bounds.add("q", bounds=models[p].bounds_from_ranges("q"), phase=p)
        x_bounds.add("qdot", bounds=models[p].bounds_from_ranges("qdot"), phase=p)
        u_bounds.add("tau", min_bound=[-50, -50], max_bound=[50, 50], phase=p)
    x_bounds[0]["q"][:, 0] = [0, Z0]  # starts at x = 0, height Z0
    x_bounds[0]["qdot"][:, 0] = [1.0, 0]  # horizontal speed 1 m/s, no vertical speed
    x_bounds[0]["q"][1, -1] = 0  # touches the floor at the end of phase 0
    x_bounds[1]["q"][0, -1] = X_END
    x_bounds[1]["qdot"][:, -1] = 0
    for p in range(2):
        u_bounds[p]["tau"][1, :] = 0  # only the horizontal force is actuated (free fall in the air)
    fcn = PhaseTransitionFcn.IMPACT if transition == "impact" else PhaseTransitionFcn.CONTINUOUS
    transitions = PhaseTransitionList()
    transitions.add(fcn, phase_pre_idx=0)
    dynamics = DynamicsOptionsList()
    dynamics.add(rk4())
    dynamics.add(rk4())
    return OptimalControlProgram(
        models,
        N_IMP,
        (T_FALL, T_SLIDE),
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objectives,
        phase_transitions=transitions,
        use_sx=False,
    )


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    want = (
        lambda name: not sys.argv[1:] or name in sys.argv[1:]
    )  # e.g. "... generate_features_data.py parameters impact"

    if want("objectives"):
        print("1. objectives")
        res = {"mayer_weights": np.array(MAYER_WEIGHTS), "n_shooting": N1, "final_time": T1}
        for i, w in enumerate(MAYER_WEIGHTS):
            res.update(extract(solve(ocp_objectives(w)), f"w{i}"))
        np.savez_compressed(OUT / "features_objectives.npz", **res)

    if want("constraints"):
        print("2. constraints")
        res = {
            "u_maxs": np.array(U_MAXS),
            "n_shooting": N1,
            "final_time": T1,
            "cart_limits": np.array(CART_LIMITS),
            "cart_limit": CART_LIMITS[-1],
        }
        free = None
        for i, u in enumerate(U_MAXS):
            sol = solve(ocp_constraints(u))
            res.update(extract(sol, f"u{i}"))
            free = free or sol  # u0 (|tau| <= 100 N, inactive) is the unbounded-position solution
        # bound on the position: continuation, free solution -> 0.9 -> 0.7 -> 0.5 (y_final stays free, as for the free run)
        previous = free
        for j, lim in enumerate(CART_LIMITS):
            previous = solve(ocp_constraints(100.0, cart_limit=lim, previous=previous))
            res.update(extract(previous, f"cart{j}"))
        res.update(
            {
                k.replace(f"cart{len(CART_LIMITS) - 1}", "cart"): v
                for k, v in res.items()
                if k.startswith(f"cart{len(CART_LIMITS) - 1}_")
            }
        )
        # too tight (not shown in the video): same continuation, one more step
        res.update(extract(solve(ocp_constraints(100.0, CART_LIMIT_TOO_TIGHT, previous)), "cart_tight"))
        np.savez_compressed(OUT / "features_constraints.npz", **res)

    if want("multiphase"):
        print("3. multiphase")
        res = {"n_shooting": np.array(N_PH), "phase_times": np.array(T_PH)}
        for tag in ("continuous", "discontinuous"):
            res.update(extract(solve(ocp_multiphase(tag)), tag))
        np.savez_compressed(OUT / "features_multiphase.npz", **res)

    if want("time"):
        print("4. free time")
        res = {"u_maxs": np.array(U_MAXS_TIME), "n_shooting": N4, "t_guess": T4_GUESS}
        for i, u in enumerate(U_MAXS_TIME):
            res.update(extract(solve(ocp_time(u)), f"u{i}"))
        np.savez_compressed(OUT / "features_time.npz", **res)

    if want("parameters"):
        print("5. parameters (continuation on the weight)")
        res = {"weights": np.array(PAR_WEIGHTS), "n_shooting": N1, "final_time": T1, "p_max": P_MAX, "p_init": P_INIT}
        previous = None
        for i, w in enumerate(PAR_WEIGHTS):
            ocp = ocp_parameters(w, previous)
            if i == 0:  # real layout of the decision vector: [dt | X | U | parameters]
                nlp = ocp.nlp[0]
                res.update(
                    layout_total=ocp.vector_layout.total_size,
                    layout_dt=ocp.dt_parameter.shape,
                    layout_x_nodes=nlp.ns + 1,
                    layout_x_per_node=nlp.states.shape,
                    layout_u_nodes=nlp.ns,
                    layout_u_per_node=nlp.controls.shape,
                    layout_params=ocp.parameters.shape,
                )
            sol = solve(ocp)
            res.update(extract(sol, f"w{i}"))
            res[f"w{i}_max_tau"] = float(sol.parameters["max_tau"][0])
            previous = sol
        np.savez_compressed(OUT / "features_parameters.npz", **res)

    if want("impact"):
        print("6. impact")
        res = {"n_shooting": np.array(N_IMP), "phase_times": np.array([T_FALL, T_SLIDE]), "z0": Z0, "gravity": G}
        for tag in ("impact", "continuous"):
            res.update(extract(solve(ocp_impact(tag)), tag))
        np.savez_compressed(OUT / "features_impact.npz", **res)
    print("done")
