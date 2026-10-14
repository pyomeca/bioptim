"""
Solve the same double-pendulum task twice (REAL bioptim / IPOPT solves) to feed ``anim_mapping.py``:
  * without mapping: two independent torques tau = [tau_1, tau_2]
  * with BiMappingList.add("tau", to_second=[0, 0], to_first=[0]): the two joints share ONE torque
Output: data/mapping_double_pendulum.npz

Usage (env with bioptim on the PATH), from the repo root:
    PYTHONPATH=. python docs/animations/generate_mapping_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    BiMappingList,
    BoundsList,
    DynamicsOptions,
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

MODEL = ExampleUtils.folder + "/models/double_pendulum.bioMod"
OUT = Path(__file__).parent / "data"
N, T, TAU_MAX = 30, 3.0, 60.0
Q_END = [1.0, 1.0]


def build(mapped: bool):
    bio_model = TorqueBiorbdModel(MODEL)
    mappings = None
    n_tau = bio_model.nb_tau
    if mapped:
        mappings = BiMappingList()
        mappings.add("tau", to_second=[0, 0], to_first=[0])
        n_tau = 1
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1.0, node=Node.ALL_SHOOTING)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"][:, 0] = 0
    x_bounds["q"][:, -1] = Q_END
    x_bounds["qdot"][:, -1] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-TAU_MAX] * n_tau, [TAU_MAX] * n_tau
    return OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=5)),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objectives,
        variable_mappings=mappings,
        use_sx=True,
    )


if __name__ == "__main__":
    out = {"n_shooting": N, "final_time": T, "q_end": np.array(Q_END)}
    for tag, mapped in (("free", False), ("mapped", True)):
        ocp = build(mapped)
        solver = Solver.IPOPT(show_online_optim=False)
        solver.set_print_level(0)
        solver.set_maximum_iterations(500)
        sol = ocp.solve(solver)
        c = sol.decision_controls()["tau"]
        tau = np.array([c[k][:, 0] for k in range(len(c))])  # (n_nodes, n_tau_optimised)
        if tau.shape[1] == 1:
            tau = np.repeat(tau, 2, axis=1)  # to_second=[0, 0]: expand to the two joints
        st = sol.decision_states()
        out[f"{tag}_tau"] = tau
        out[f"{tag}_q"] = np.array([st["q"][k][:, 0] for k in range(N + 1)])
        out[f"{tag}_cost"] = float(sol.cost)
        out[f"{tag}_status"] = int(sol.status)
        out[f"{tag}_iterations"] = int(sol.iterations)
        out[f"{tag}_n_vars"] = int(sol.vector.shape[0])
        out[f"{tag}_n_tau_vars"] = int(np.prod(c[0].shape) * len(c)) if not mapped else len(c)
        print(
            tag,
            "status",
            sol.status,
            "iters",
            sol.iterations,
            "cost",
            sol.cost,
            "nvars",
            sol.vector.shape[0],
            "tau shape",
            np.array([c[k] for k in range(len(c))]).shape,
            "qend",
            out[f"{tag}_q"][-1],
        )
    np.savez(OUT / "mapping_double_pendulum.npz", **out)
