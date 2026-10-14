"""
Architecture of Bioptim: sizes read from a REAL tiny OCP (pendulum, one phase, N = 20 intervals, T = 1 s, RK4, IPOPT).
Stored in ``data/arch_sizes.npz``.  Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_arch_data.py
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
    SolutionMerge,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
ROT, T, N = 1, 1.0, 20


def main():
    bio_model = TorqueBiorbdModel(MODEL)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][ROT, -1] = 1.0
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * bio_model.nb_tau, [100] * bio_model.nb_tau
    u_bounds["tau"][ROT, :] = 0
    x_init = InitialGuessList()
    x_init.add("q", np.array([[0, 0], [0, 1.0]]), interpolation=InterpolationType.LINEAR)
    ocp = OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4()),
        x_bounds=x_bounds,
        x_init=x_init,
        u_bounds=u_bounds,
        objective_functions=Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"),
        use_sx=True,
    )
    nlp = ocp.nlp[0]
    solver = Solver.IPOPT()
    solver.set_print_level(0)
    sol = ocp.solve(solver)
    v = ocp.nlp[0]
    print("phases", ocp.n_phases, "nlp", len(ocp.nlp), type(nlp).__name__)
    print(
        "states", nlp.states.shape, list(nlp.states.keys()), "controls", nlp.controls.shape, list(nlp.controls.keys())
    )
    print("control_type", nlp.control_type, "n_states_nodes", nlp.n_states_nodes, "ns", nlp.ns)
    print(
        "params",
        ocp.parameters.shape if hasattr(ocp.parameters, "shape") else None,
        "algebraic",
        nlp.algebraic_states.shape,
    )
    print("vector", np.array(sol.vector).size, "status", sol.status, "cost", float(sol.cost))
    print(
        "g_internal", len(nlp.g_internal), "J", len(nlp.J), "g", len(nlp.g), [type(g).__name__ for g in nlp.g_internal]
    )
    ov = ocp.ocp_solver.get_optimized_value()
    n_g = int(np.array(ov["lam_g"]).size)
    print("n_g", n_g, list(ov.keys()))
    time, st, co, al, pa = ocp.get_decision_variables()
    n_t = int(np.prod(np.shape(time))) if time is not None else 0
    n_x = sum(int(c.shape[0] * c.shape[1]) for ph in st for c in ph)
    n_u = sum(int(c.shape[0] * c.shape[1]) for ph in co for c in ph)
    print("blocks t, x, u:", n_t, n_x, n_u)
    np.savez(
        OUT / "arch_sizes.npz",
        n_phases=ocp.n_phases,
        n_shooting=N,
        nx=int(nlp.states.shape),
        nu=int(nlp.controls.shape),
        n_variables=int(np.array(sol.vector).size),
        status=int(sol.status),
        cost=float(sol.cost),
        n_g=n_g,
        n_time=n_t,
        n_x=n_x,
        n_u=n_u,
        n_obj=len(nlp.J),
        n_constraints=len(nlp.g),
        n_internal=len(nlp.g_internal),
    )


if __name__ == "__main__":
    main()
