"""
Real stochastic optimal control of a 2-link arm reaching (Leuven arm model, torque driven), driven by REAL bioptim/IPOPT
solves stored in data/socpk_results.npz.  The problem is the bioptim example
bioptim/examples/toy_examples/stochastic_optimal_control/arm_reaching_torque_driven_collocations.py, called through its
own ``prepare_socp`` (unchanged), with SocpType.COLLOCATION (Legendre, degree 3), StochasticTorqueBiorbdModel, motor noise
and sensory noise on the hand position and velocity, and a feedback gain K (2 torques x 4 sensory references) that is a
CONTROL of the OCP (key "k"), like the covariance P (key "cov", 4x4 on [q, qdot]).
Three real solves are stored:
  * ``low``  : sensory noise of the example (std 3e-4 m on the hand position, 2.4e-3 m/s on the hand velocity),
  * ``high`` : the same problem with the sensory noise std multiplied by SENS_HIGH (the motor noise is unchanged),
  * ``det``  : the deterministic OCP (plain OptimalControlProgram, TorqueBiorbdModel, same cost and constraints, no
               noise, no feedback), only used as a dashed reference of the hand path.
No warm start (every solve starts from the example's own initial guess).  Tried but NOT stored: sensory noise x5 and
x10 end with IPOPT status 1 (restoration failed) after 320 and 415 iterations, so the 'high' case is x3.  Noise magnitudes are given to bioptim as
variances divided by dt (magnitude = std**2 / dt), as in the example.
The hand position is the marker 2 of the model; the hand covariance is J P_qq J' with J the Jacobian of the marker with
respect to q (computed here with CasADi from the model function).

Run from the repo root:  PYTHONPATH=. python docs/animations/generate_socpk_data.py
"""

import sys
import time
from pathlib import Path

import casadi as cas
import numpy as np
from bioptim import (
    Axis,
    ConstraintFcn,
    ConstraintList,
    ControlType,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    SolutionMerge,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.toy_examples.stochastic_optimal_control import arm_reaching_torque_driven_collocations as ex
from bioptim.examples.utils import ExampleUtils

OUT = Path(__file__).parent / "data"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SENS_HIGH = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
FINAL_TIME = 0.8
DT = FINAL_TIME / N
POLY = 3
MODEL = ExampleUtils.folder + "/models/LeuvenArmModel.bioMod"
HAND_TARGET = np.array([9.359873986980460e-12, 0.527332023564034])
Q0 = np.array([0.349065850398866, 2.245867726451909])
Q1 = np.array([0.959931088596881, 1.159394851847144])
END_BOUND_M = 0.004  # bound on the hand position std at the last node, set in the example (max_bound = 0.004**2)
TRIED_FAILED = (5.0, 10.0)
MOTOR_STD, WPQ_STD, WPQDOT_STD = 0.05, 3e-4, 0.0024


def noises(sens):
    motor = cas.DM(np.array([MOTOR_STD**2 / DT] * 2))
    pos = cas.DM(np.array([(sens * WPQ_STD) ** 2 / DT] * 2))
    vel = cas.DM(np.array([(sens * WPQDOT_STD) ** 2 / DT] * 2))
    return motor, cas.vertcat(pos, vel)


def ipopt():
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_linear_solver("mumps")
    solver.set_tol(1e-3)
    solver.set_dual_inf_tol(3e-4)
    solver.set_constr_viol_tol(1e-7)
    solver.set_maximum_iterations(2000)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    solver.set_nlp_scaling_method("none")
    return solver


def hand_path(model, q):
    hand = model.marker(2)
    return np.array([np.array(hand(q[:, i], [])).ravel()[:2] for i in range(q.shape[1])]).T


INITIAL_COV = np.diag([1e-4, 1e-4, 1e-7, 1e-7])  # same P0 as the example (its "initial_cov")
_StochasticOCP = ex.StochasticOptimalControlProgram


def socp_with_fixed_p0(*args, constraints=None, **kwargs):
    """
    The example leaves the covariance control "cov" of the FIRST node free (bounds +-inf): IPOPT then returns a
    non-physical P (negative variances).  This wrapper adds the missing constraint P(t=0) = initial_cov and calls the
    unchanged StochasticOptimalControlProgram; everything else is the example's prepare_socp.
    """
    constraints.add(
        ConstraintFcn.TRACK_CONTROL,
        key="cov",
        node=Node.START,
        target=INITIAL_COV.reshape(-1, order="F"),
    )
    return _StochasticOCP(*args, constraints=constraints, **kwargs)


ex.StochasticOptimalControlProgram = socp_with_fixed_p0


def solve_socp(sens):
    motor, sensory = noises(sens)
    socp = ex.prepare_socp(
        biorbd_model_path=MODEL,
        final_time=FINAL_TIME,
        n_shooting=N,
        polynomial_degree=POLY,
        hand_final_position=HAND_TARGET,
        motor_noise_magnitude=motor,
        sensory_noise_magnitude=sensory,
        use_sx=True,
    )
    t0 = time.time()
    sol = socp.solve(ipopt())
    wall = time.time() - t0
    st = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    ct = sol.stepwise_controls(to_merge=SolutionMerge.NODES)
    q_all = np.array(st["q"])  # POLY + 2 columns per interval (node, collocation points, ...) + last node
    q = q_all[:, :: POLY + 2]  # the N + 1 shooting nodes
    return socp, dict(
        q=q,
        q_all=q_all,
        qdot=np.array(st["qdot"]),
        tau=np.array(ct["tau"]),
        k=np.array(ct["k"]),
        cov=np.array(ct["cov"]),
        status=int(sol.status),
        iterations=int(sol.iterations),
        seconds=wall,
        cost=float(np.ravel(sol.cost)[0]),
        hand=hand_path(socp.nlp[0].model, q),
        hand_all=hand_path(socp.nlp[0].model, q_all),
    )


def solve_det():
    bio_model = TorqueBiorbdModel(MODEL)
    bio_model.set_friction_coefficients(np.array([[0.05, 0.025], [0.025, 0.05]]))
    objectives = ObjectiveList()
    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=1e3 / 2, quadratic=True
    )
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", node=Node.START, target=Q0)
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.START, target=np.array([0, 0]))
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.END, target=np.array([0, 0]))
    constraints.add(
        ConstraintFcn.TRACK_MARKERS, node=Node.END, target=HAND_TARGET, marker_index=2, axes=[Axis.X, Axis.Y]
    )
    x_init = InitialGuessList()
    q_init = np.array([np.linspace(Q0[0], Q1[0], N + 1), np.linspace(Q0[1], Q1[1], N + 1)])
    x_init.add("q", initial_guess=q_init, interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", initial_guess=np.zeros((2, N + 1)), interpolation=InterpolationType.EACH_FRAME)
    ocp = OptimalControlProgram(
        bio_model,
        N,
        FINAL_TIME,
        dynamics=DynamicsOptions(
            ode_solver=OdeSolver.COLLOCATION(polynomial_degree=POLY, method="legendre"), expand_dynamics=True
        ),
        x_init=x_init,
        objective_functions=objectives,
        constraints=constraints,
        control_type=ControlType.CONSTANT_WITH_LAST_NODE,
        use_sx=True,
    )
    t0 = time.time()
    sol = ocp.solve(ipopt())
    wall = time.time() - t0
    q_all = np.array(sol.stepwise_states(to_merge=SolutionMerge.NODES)["q"])
    q = q_all[:, :: POLY + 1]  # a plain OCP has POLY + 1 columns per interval + last node
    return dict(
        q=q,
        q_all=q_all,
        status=int(sol.status),
        iterations=int(sol.iterations),
        seconds=wall,
        hand=hand_path(ocp.nlp[0].model, q),
        hand_all=hand_path(ocp.nlp[0].model, q_all),
    )


def hand_cov(socp, q, cov):
    """2x2 covariance of the hand position at every node: J P_qq J', J = d(marker 2)/dq (CasADi)."""
    qs = cas.MX.sym("q", 2)
    jac = cas.Function("J", [qs], [cas.jacobian(socp.nlp[0].model.marker(2)(qs, [])[:2], qs)])
    out = np.zeros((q.shape[1], 2, 2))
    for i in range(q.shape[1]):
        P = cov[:, i].reshape(4, 4, order="F")
        J = np.array(jac(q[:, i]))
        out[i] = J @ P[:2, :2] @ J.T
    return out


if __name__ == "__main__":
    res = {}
    r = solve_det()
    print("RESULT det status", r["status"], "iters", r["iterations"], "wall", round(r["seconds"], 1), flush=True)
    res.update({f"det_{k}": v for k, v in r.items()})
    for name, sens in (("low", 1.0), ("high", SENS_HIGH)):
        socp, r = solve_socp(sens)
        r["hand_cov"] = hand_cov(socp, r["q"], r["cov"])
        r["sens"] = sens
        print(
            "RESULT",
            name,
            "status",
            r["status"],
            "iters",
            r["iterations"],
            "wall",
            round(r["seconds"], 1),
            "cost",
            r["cost"],
            flush=True,
        )
        res.update({f"{name}_{k}": v for k, v in r.items()})
    # sensory noise multipliers that did NOT converge: only the IPOPT status and iterations are stored (no curves)
    tried = []
    for sens in TRIED_FAILED:
        _, r = solve_socp(sens)
        print(
            "TRIED sens",
            sens,
            "status",
            r["status"],
            "iters",
            r["iterations"],
            "wall",
            round(r["seconds"], 1),
            flush=True,
        )
        tried.append((sens, r["status"], r["iterations"]))
    res.update(
        tried_sens=[t[0] for t in tried], tried_status=[t[1] for t in tried], tried_iterations=[t[2] for t in tried]
    )
    res.update(
        end_bound_m=END_BOUND_M,
        n=N,
        poly=POLY,
        final_time=FINAL_TIME,
        wpq_std=WPQ_STD,
        wpqdot_std=WPQDOT_STD,
        motor_std=MOTOR_STD,
    )
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "socpk_results.npz", **res)
    for name in ("low", "high"):
        cov = res[f"{name}_hand_cov"]
        print(
            name,
            "K shape",
            res[f"{name}_k"].shape,
            "|K| max",
            np.abs(res[f"{name}_k"]).max(),
            "end hand std (mm)",
            np.sqrt(np.diag(cov[-1])) * 1e3,
            "max std (mm)",
            np.sqrt(cov[:, [0, 1], [0, 1]]).max() * 1e3,
        )
    print("hand start", res["low_hand"][:, 0], "end", res["low_hand"][:, -1])
    print("max |det hand - low hand| (mm)", np.abs(res["det_hand"] - res["low_hand"]).max() * 1e3)
    print("max |high hand - low hand| (mm)", np.abs(res["high_hand"] - res["low_hand"]).max() * 1e3)
    for name in ("low", "high"):
        k = res[f"{name}_k"]
        std = np.sqrt(np.array([np.linalg.eigvalsh(c).max() for c in res[f"{name}_hand_cov"]])) * 1e3
        print(
            name,
            "mean|K|",
            np.abs(k).mean(),
            "max|K|",
            np.abs(k).max(),
            "peak major std (mm)",
            std.max(),
            "cost",
            res[f"{name}_cost"],
        )
