"""
Solve the bioptim pendulum swing-up OCP twice (Direct Multiple Shooting with RK4, Direct Collocation) and save the
results to ``data/pendulum_solutions.npz`` for the Manim scenes in ``dms_vs_dc.py``.

The OCP is the one of bioptim/examples/toy_examples/sqp_method/pendulum.py (same as the README tutorial): a pendulum
starts hanging down (q = 0) and must end upward (q = 3.14) at rest, with the torque only on the rotation (sideways
translation is passive), minimizing the integral of tau^2 (ObjectiveFcn.Lagrange.MINIMIZE_CONTROL).

Everything saved here is REAL bioptim output (no fallback involved).

Usage (from an environment with bioptim, biorbd, casadi):
    python docs/animations/generate_pendulum_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    Objective,
    ObjectiveFcn,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

N_SHOOTING = 20
FINAL_TIME = 1.0
POLYNOMIAL_DEGREE = 3
N_RK4_STEPS = 5
EARLY_ITERATION = {"rk4": 0, "col": 0}  # IPOPT iteration at which the iterate is stored


def prepare_ocp(ode_solver) -> OptimalControlProgram:
    """Same OCP as the bioptim pendulum example, only the OdeSolver changes."""
    bio_model = TorqueBiorbdModel(ExampleUtils.folder + "/models/pendulum.bioMod")

    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    dynamics = DynamicsOptions(ode_solver=ode_solver)

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 3.14
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * bio_model.nb_tau, [100] * bio_model.nb_tau
    u_bounds["tau"][1, :] = 0

    # Initial guess: q from hanging to upright and qdot from 0 to 3.14 linearly in time, zero controls. Only used to start IPOPT (it is what we draw
    # as the "not converged" iterate: the integrated segments do not reach the next node yet).
    x_init = InitialGuessList()
    x_init.add("q", [[0, 0], [0, 3.14]], interpolation=InterpolationType.LINEAR)
    x_init.add("qdot", [[0, 0], [0, 3.14]], interpolation=InterpolationType.LINEAR)

    return OptimalControlProgram(
        bio_model,
        N_SHOOTING,
        FINAL_TIME,
        dynamics=dynamics,
        x_init=x_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        use_sx=True,
    )


def solve(ode_solver, tag: str, max_iter: int = None):
    """
    Solve and return the arrays to store, prefixed by ``tag``.

    Shapes (N = n_shooting, dof = 2: [translation y, rotation x]):
      q_nodes, qdot_nodes : (dof, N+1)   states at the shooting nodes
      q_steps, qdot_steps : (N, dof, m)  states inside each interval: RK4 -> m = n_integration_steps + 1 points of the
                                         integration (the last one is F(x_k, u_k)); collocation -> m = degree + 1 points
                                         (node + collocation points, the polynomial values)
      t_steps             : (N, m)       time of these points
      tau                 : (dof, N)     piecewise-constant controls
    """
    ocp = prepare_ocp(ode_solver)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    if max_iter is not None:
        solver.set_maximum_iterations(max_iter)
    sol = ocp.solve(solver)

    n = N_SHOOTING
    step_states = sol.stepwise_states()
    step_time = sol.stepwise_time()
    out = {
        f"{tag}_q_nodes": np.array([sol.decision_states()["q"][k][:, 0] for k in range(n + 1)]).T,
        f"{tag}_qdot_nodes": np.array([sol.decision_states()["qdot"][k][:, 0] for k in range(n + 1)]).T,
        f"{tag}_tau": np.array([sol.decision_controls()["tau"][k][:, 0] for k in range(n)]).T,
        f"{tag}_q_steps": np.array([step_states["q"][k] for k in range(n)]),
        f"{tag}_qdot_steps": np.array([step_states["qdot"][k] for k in range(n)]),
        f"{tag}_t_steps": np.array([np.array(step_time[k]).ravel() for k in range(n)]),
        f"{tag}_n_decision_variables": int(sol.vector.shape[0]),
        f"{tag}_cost": float(sol.cost),
        f"{tag}_iterations": int(sol.iterations),
    }
    return out


if __name__ == "__main__":
    results = {"n_shooting": N_SHOOTING, "final_time": FINAL_TIME, "polynomial_degree": POLYNOMIAL_DEGREE}
    rk4 = OdeSolver.RK4(n_integration_steps=N_RK4_STEPS)
    col = OdeSolver.COLLOCATION(polynomial_degree=POLYNOMIAL_DEGREE, method="legendre")
    # "rk4" / "col": converged solutions; "rk4_it" / "col_it": an early (NOT converged) IPOPT iterate, in which the
    # continuity constraints are still violated -> real gaps between the integrated segments and the next node.
    for tag, ode, max_iter in (
        ("rk4", rk4, None),
        ("col", col, None),
        ("rk4_it", rk4, EARLY_ITERATION["rk4"]),
        ("col_it", col, EARLY_ITERATION["col"]),
    ):
        out = solve(ode, tag, max_iter)
        results.update(out)
        print(tag, {k[len(tag) + 1 :]: v for k, v in out.items() if np.isscalar(v)})
    path = Path(__file__).parent / "data" / "pendulum_solutions.npz"
    np.savez_compressed(path, **results)
    print("saved", path)
