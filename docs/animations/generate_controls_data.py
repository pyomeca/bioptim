"""
Solve the same pendulum swing-up with three control types (REAL bioptim / IPOPT solves) to feed ``anim_controls.py``:
ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS and ControlType.CONSTANT_WITH_LAST_NODE.
Output: data/controls_types.npz

Usage (env with bioptim on the PATH), from the repo root:
    PYTHONPATH=. python docs/animations/generate_controls_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    ControlType,
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

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
ROT, N, T, TAU_MAX = 1, 20, 1.0, 100.0
TYPES = {
    "constant": ControlType.CONSTANT,
    "linear": ControlType.LINEAR_CONTINUOUS,
    "last": ControlType.CONSTANT_WITH_LAST_NODE,
}


def build(control_type):
    bio_model = TorqueBiorbdModel(MODEL)
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1.0, node=Node.ALL_SHOOTING)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"][:, 0] = 0
    x_bounds["q"][ROT, -1] = 3.14
    x_bounds["qdot"][:, -1] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-TAU_MAX] * bio_model.nb_tau, [TAU_MAX] * bio_model.nb_tau
    u_bounds["tau"][ROT, :] = 0
    return OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=5)),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objectives,
        control_type=control_type,
        use_sx=True,
    )


if __name__ == "__main__":
    out = {"n_shooting": N, "final_time": T}
    for tag, ct in TYPES.items():
        ocp = build(ct)
        solver = Solver.IPOPT(show_online_optim=False)
        solver.set_print_level(0)
        solver.set_maximum_iterations(500)
        sol = ocp.solve(solver)
        t = np.array(sol.decision_time(to_merge=SolutionMerge.NODES)).ravel()
        st, ct_ = sol.decision_states(), sol.decision_controls()
        tau = np.array(
            [ct_["tau"][k][0, 0] for k in range(len(ct_["tau"]))]
        )  # value u_k at each node holding a control
        ncols = [ct_["tau"][k].shape[1] for k in range(len(ct_["tau"]))]
        out[f"{tag}_t"] = t
        out[f"{tag}_tau_raw"] = tau
        out[f"{tag}_ncols"] = np.array(ncols)
        out[f"{tag}_theta"] = np.array([st["q"][k][ROT, 0] for k in range(N + 1)])
        out[f"{tag}_cost"] = float(sol.cost)
        out[f"{tag}_iterations"] = int(sol.iterations)
        out[f"{tag}_status"] = int(sol.status)
        out[f"{tag}_n_vars"] = (
            int(ocp.variables_vector.shape[0]) if hasattr(ocp, "variables_vector") else int(sol.vector.shape[0])
        )
        out[f"{tag}_n_vars_sol"] = int(sol.vector.shape[0])
        out[f"{tag}_n_ctrl_nodes"] = len(ct_["tau"])
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
            "ncols",
            ncols[:3],
            ncols[-2:],
            "nodes",
            len(ct_["tau"]),
        )
    np.savez(OUT / "controls_types.npz", **out)
