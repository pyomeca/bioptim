"""
MultiBiorbdModel: two independent bodies optimised in ONE OCP. Everything stored in ``data/multibody_*.npz`` is REAL
bioptim / IPOPT output.

Body A (``models/multibody_a.bioMod``): one pendulum link, 1 DoF, pivot at y = -0.8, length 1.0.
Body B (``models/multibody_b.bioMod``): double pendulum, 2 DoF, pivot at y = +0.8, links of 0.6 + 0.6.
Both start hanging at rest.  Nothing couples them in the dynamics (block-diagonal), only a Constraint at the last
node superimposes the tip of A on the tip of B (SUPERIMPOSE_MARKERS on global marker names).

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_multibody_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    ConstraintFcn,
    ConstraintList,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    MultiTorqueBiorbdModel,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    SolutionMerge,
)

HERE = Path(__file__).parent
MODELS = (str(HERE / "models" / "multibody_a.bioMod"), str(HERE / "models" / "multibody_b.bioMod"))
T, N = 1.5, 30


def prepare_ocp():
    bio_model = MultiTorqueBiorbdModel(MODELS)
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="A_tip", second_marker="B_tip")
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1)

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"][:, 0] = 0
    x_bounds["qdot"][:, -1] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-60] * bio_model.nb_tau, [60] * bio_model.nb_tau
    x_init = InitialGuessList()
    x_init.add("q", np.array([[0, 0.9], [0, -1.0], [0, 0.6]]), interpolation=InterpolationType.LINEAR)
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
        use_sx=True,
    )
    return ocp, bio_model


def main():
    ocp, bio_model = prepare_ocp()
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(500)
    sol = ocp.solve(solver)
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = np.array(states["q"]), np.array(states["qdot"]), np.array(controls["tau"])
    print("status", sol.status, "iters", sol.iterations, "cost", float(sol.cost))
    print("keys", list(states.keys()), q.shape, qdot.shape, tau.shape)
    print("nb_q", bio_model.nb_q, "variable_index q", [list(bio_model.variable_index("q", i)) for i in range(2)])
    print("variable_index markers", [list(bio_model.variable_index("markers", i)) for i in range(2)])
    print("marker_names", bio_model.marker_names)
    mk = bio_model.markers()
    tips = np.array([np.array(mk(q[:, k], [])).T for k in range(q.shape[1])])  # (N+1, 3 markers, 3 xyz)
    gap = np.linalg.norm(tips[:, 0, :] - tips[:, 2, :], axis=1)
    print("gap start/end", gap[0], gap[-1])
    np.savez(
        HERE / "data" / "multibody_results.npz",
        t=np.linspace(0, T, N + 1),
        q=q,
        qdot=qdot,
        tau=tau,
        markers=tips,
        gap=gap,
        idx_q0=np.array(bio_model.variable_index("q", 0)),
        idx_q1=np.array(bio_model.variable_index("q", 1)),
        idx_marker0=np.array(bio_model.variable_index("markers", 0)),
        idx_marker1=np.array(bio_model.variable_index("markers", 1)),
        marker_names=np.array(bio_model.marker_names),
        status=int(sol.status),
        iterations=int(sol.iterations),
        cost=float(sol.cost),
        n=N,
        T=T,
    )


if __name__ == "__main__":
    main()
