"""
Custom objective and custom constraint: a user function ``f(controller, **extra)`` returning a casadi expression is
passed in place of a ``ObjectiveFcn`` / ``ConstraintFcn``. Everything stored in ``data/custompen_pendulum.npz`` is REAL
bioptim / IPOPT output (one-link pendulum ``models/custompen_pendulum.bioMod``: 1 kg point mass at the tip of a 1 m
rod, one torque at the pivot, q = 0 hanging, q = pi upright; swing from q = 0 to q = pi, at rest at both ends,
N = 30 intervals, T = 2 s, RK4, torque bounds +-TAU_MAX).

Three independent solves, same linear initial guess (no warm start, no continuation):
  plain   MINIMIZE_CONTROL (tau), weight W_TAU
  obj     plain + custom Lagrange objective ``tip_height`` (height of the tip marker, read through controller.model),
          weight W_HEIGHT, quadratic=False (the cost is the integral of the tip height: it pushes the tip down)
  con     plain + custom constraint ``power`` (tau * qdot, read from controller.controls and controller.states) kept
          between -P_MAX and +P_MAX at every shooting node; P_MAX = FRACTION x peak |power| of the plain solve
A non-convex problem: each solve returns A local minimum (nothing here proves it is the global one).

Stored: t (N+1), q, qdot, tau (N+1, last value repeated for plotting), tip height z, power tau*qdot, and per run
status / iterations / cost / converged, plus the settings.

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_custompen_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    ConstraintList,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    PenaltyController,
    Solver,
    SolutionMerge,
    TorqueBiorbdModel,
)
from casadi import MX

HERE = Path(__file__).parent
MODEL = HERE / "models" / "custompen_pendulum.bioMod"
OUT = HERE / "data"
N, T = 30, 2.0
TAU_MAX = 40.0
W_TAU = 1e-2
W_HEIGHT = 1.0  # must equal the literal weight=1.0 of the custom objective in prepare_ocp (checked below)
FRACTION = 0.6


def tip_height(controller: PenaltyController, marker: str) -> MX:
    q = controller.states["q"].cx
    tip = controller.model.markers()(q, controller.parameters.cx)
    return tip[2, controller.model.marker_index(marker)]


def power(controller: PenaltyController) -> MX:
    return controller.controls["tau"].cx * controller.states["qdot"].cx


def prepare_ocp(kind, p_max=None):
    bio_model = TorqueBiorbdModel(str(MODEL))
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=W_TAU)
    constraints = ConstraintList()
    if kind == "obj":
        objectives.add(tip_height, custom_type=ObjectiveFcn.Lagrange, weight=1.0, quadratic=False, marker="tip")
    if kind == "con":
        constraints.add(power, node=Node.ALL_SHOOTING, min_bound=-p_max, max_bound=p_max)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][0, 0] = 0.0
    x_bounds["q"][0, -1] = np.pi
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0.0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-TAU_MAX], [TAU_MAX]
    x_init = InitialGuessList()
    x_init.add("q", np.array([[0.0, np.pi]]), interpolation=InterpolationType.LINEAR)
    ocp = OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=5)),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objectives,
        constraints=constraints,
        use_sx=False,
    )
    if kind == "obj":
        assert np.all(np.asarray(ocp.nlp[0].J[-1].weight) == W_HEIGHT)
    return ocp


def solve(kind, p_max=None):
    ocp = prepare_ocp(kind, p_max)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(500)
    sol = ocp.solve(solver)
    q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
    qd = sol.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
    tau = sol.decision_controls(to_merge=SolutionMerge.NODES)["tau"]
    tau = np.hstack([tau, tau[:, -1:]]) if tau.shape[1] == q.shape[1] - 1 else tau
    return dict(
        t=np.linspace(0, T, q.shape[1]),
        q=q[0],
        qdot=qd[0],
        tau=tau[0],
        z=-np.cos(q[0]),
        power=tau[0] * qd[0],
        status=int(sol.status),
        iterations=int(sol.iterations),
        cost=float(sol.cost),
        converged=bool(sol.status == 0),
    )


def main():
    out = {}
    plain = solve("plain")
    p_max = FRACTION * float(np.abs(plain["power"][:-1]).max())
    runs = {"plain": plain, "obj": solve("obj"), "con": solve("con", p_max)}
    for name, r in runs.items():
        print(
            f"{name}: status={r['status']} it={r['iterations']} cost={r['cost']:.4g} peak|tau|={np.abs(r['tau']).max():.2f}"
            f" peak|P|={np.abs(r['power'][:-1]).max():.2f} mean z={r['z'].mean():.3f}"
        )
        for key, v in r.items():
            out[f"{name}_{key}"] = v
    print(f"P_MAX = {p_max:.2f} W")
    out.update(
        n_shooting=N, final_time=T, tau_max=TAU_MAX, w_tau=W_TAU, w_height=W_HEIGHT, p_max=p_max, fraction=FRACTION
    )
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "custompen_pendulum.npz", **out)


if __name__ == "__main__":
    main()
