"""
Reading the decision vector. A small REAL Bioptim OCP (pendulum, rotation 0 -> 1 rad, sliding translation actuated,
N = 3 shooting intervals, T = 1 s, one parameter "max_tau") is built four times: RK4 or COLLOCATION(degree 3), each with
``OrderingStrategy.VARIABLE_MAJOR`` and ``OrderingStrategy.TIME_MAJOR``. Everything stored in ``data/vector_layout.npz``
is read from the real objects:
    ocp.vector_layout.index_map   {(phase, var_type, node) | ("global", "time"/"parameters"): (slice, n_columns)}
    OptimizationVectorHelper.bounds_vectors(ocp) / init_vector(ocp)   the bounds and initial guess vectors
    sol.vector                    the optimal decision vector returned by IPOPT
    sol.decision_states / decision_controls / parameters   the recommended way to read it
The collocation case is warm-started from nothing special (linear interpolation); IPOPT status is stored.

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_vector_data.py
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
    Objective,
    ObjectiveFcn,
    OdeSolver,
    OptimalControlProgram,
    OrderingStrategy,
    ParameterList,
    ParameterObjectiveList,
    Solver,
    TorqueBiorbdModel,
    VariableScaling,
)
from bioptim.examples.utils import ExampleUtils
from bioptim.optimization.optimization_vector import OptimizationVectorHelper

MODEL = ExampleUtils.folder + "/models/pendulum.bioMod"
OUT = Path(__file__).parent / "data"
ROT, N, T, TARGET = 1, 3, 1.0, 1.0


def no_model_change(bio_model, parameter):
    return


def max_tau_upper(controller):
    return controller.parameters["max_tau"].cx - controller.controls["tau"].cx[0]


def max_tau_lower(controller):
    return controller.parameters["max_tau"].cx + controller.controls["tau"].cx[0]


def prepare_ocp(ode_solver, ordering, warm=None):
    parameters = ParameterList(use_sx=True)
    parameters.add("max_tau", no_model_change, size=1, scaling=VariableScaling("max_tau", [1]))
    parameter_bounds = BoundsList()
    parameter_bounds.add("max_tau", min_bound=0, max_bound=100, interpolation=InterpolationType.CONSTANT)
    parameter_init = InitialGuessList()
    parameter_init["max_tau"] = 30.0
    parameter_objectives = ParameterObjectiveList()
    parameter_objectives.add(ObjectiveFcn.Parameter.MINIMIZE_PARAMETER, key="max_tau", weight=1e-3, quadratic=True)
    constraints = ConstraintList()
    constraints.add(max_tau_upper, node=Node.ALL_SHOOTING, min_bound=0, max_bound=np.inf)
    constraints.add(max_tau_lower, node=Node.ALL_SHOOTING, min_bound=0, max_bound=np.inf)

    bio_model = TorqueBiorbdModel(MODEL, parameters=parameters)
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0
    x_bounds["q"][ROT, -1] = TARGET
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * bio_model.nb_tau, [100] * bio_model.nb_tau
    u_bounds["tau"][ROT, :] = 0
    x_init = InitialGuessList()
    u_init = InitialGuessList()
    if warm is None:
        x_init.add("q", np.array([[0, 0], [0, TARGET]]), interpolation=InterpolationType.LINEAR)
    else:  # warm start from the collocation solution at the shooting nodes
        each = InterpolationType.EACH_FRAME
        x_init.add("q", warm["q"], interpolation=each)
        x_init.add("qdot", warm["qdot"], interpolation=each)
        u_init.add("tau", warm["tau"], interpolation=each)
        parameter_init["max_tau"] = warm["max_tau"]
    return OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=ode_solver),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"),
        constraints=constraints,
        parameters=parameters,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        parameter_objectives=parameter_objectives,
        ordering_strategy=ordering,
        use_sx=True,
    )


CASES = {
    "colloc_var": (lambda: OdeSolver.COLLOCATION(polynomial_degree=3), OrderingStrategy.VARIABLE_MAJOR),
    "colloc_time": (lambda: OdeSolver.COLLOCATION(polynomial_degree=3), OrderingStrategy.TIME_MAJOR),
    "rk4_var": (lambda: OdeSolver.RK4(n_integration_steps=4), OrderingStrategy.VARIABLE_MAJOR),
    "rk4_time": (lambda: OdeSolver.RK4(n_integration_steps=4), OrderingStrategy.TIME_MAJOR),
}


def label(key):
    if key[0] == "global":
        return key[1]
    return f"{key[1]}:{key[2]}"


def main():
    res = {}
    warm = None
    for tag, (ode, ordering) in CASES.items():
        ocp = prepare_ocp(ode(), ordering, warm if tag.startswith("rk4") else None)
        layout = ocp.vector_layout
        keys = list(layout.index_map.keys())
        res[f"{tag}_keys"] = np.array([label(k) for k in keys])
        res[f"{tag}_start"] = np.array([layout.index_map[k][0].start for k in keys])
        res[f"{tag}_stop"] = np.array([layout.index_map[k][0].stop for k in keys])
        res[f"{tag}_ncols"] = np.array([layout.index_map[k][1] for k in keys])
        res[f"{tag}_total"] = layout.total_size
        v_min, v_max = OptimizationVectorHelper.bounds_vectors(ocp)
        res[f"{tag}_lbx"] = np.array(v_min).ravel()
        res[f"{tag}_ubx"] = np.array(v_max).ravel()
        res[f"{tag}_init"] = np.array(OptimizationVectorHelper.init_vector(ocp)).ravel()
        solver = Solver.IPOPT(show_online_optim=False)
        solver.set_print_level(0)
        solver.set_maximum_iterations(500)
        sol = ocp.solve(solver)
        vec = np.array(sol.vector).ravel()
        res[f"{tag}_x"] = vec
        res[f"{tag}_status"] = int(sol.status)
        res[f"{tag}_iterations"] = int(sol.iterations)
        res[f"{tag}_cost"] = float(sol.cost)
        # recommended reading, to cross-check the layout: decision_states / decision_controls / parameters
        st, co = sol.decision_states(), sol.decision_controls()
        res[f"{tag}_q_nodes"] = np.array([np.array(st["q"][k])[:, 0] for k in range(N + 1)]).T
        res[f"{tag}_tau"] = np.array([np.array(co["tau"][k])[:, 0] for k in range(N)]).T
        res[f"{tag}_max_tau"] = float(np.array(sol.parameters["max_tau"]).ravel()[0])
        # the layout can be unstacked back: cross-check with the recommended reading (decision_states)
        un = layout.unstack(vec.reshape(-1, 1))
        q1_raw = un[(0, "states", 1)][:2, :]  # (n_q, n_columns) rows q, columns node + collocation points
        q1_api = np.array(st["q"][1])
        assert q1_raw.shape == q1_api.shape and np.allclose(q1_raw, q1_api), (q1_raw, q1_api)
        res[f"{tag}_q1"] = q1_api
        res[f"{tag}_n_states_decision_steps_1"] = int(ocp.nlp[0].n_states_decision_steps(1))
        res[f"{tag}_x1_block"] = un[(0, "states", 1)]
        if tag == "colloc_var":
            warm = {
                "q": res[f"{tag}_q_nodes"],
                "qdot": np.array([np.array(st["qdot"][k])[:, 0] for k in range(N + 1)]).T,
                "tau": res[f"{tag}_tau"],
                "max_tau": res[f"{tag}_max_tau"],
            }
        print(tag, "total", layout.total_size, "status", sol.status, "iters", sol.iterations, "cost", sol.cost)
        for k in keys:
            print("   ", k, layout.index_map[k])
        print("   q_nodes", res[f"{tag}_q_nodes"][ROT], "tau", res[f"{tag}_tau"][0], "max_tau", res[f"{tag}_max_tau"])
    # same solution in both orderings: the vector is a permutation of the other
    for a, b in (("rk4_var", "rk4_time"), ("colloc_var", "colloc_time")):
        print(a, b, "sorted vectors max diff", np.abs(np.sort(res[f"{a}_x"]) - np.sort(res[f"{b}_x"])).max())
    np.savez(OUT / "vector_layout.npz", **res)


if __name__ == "__main__":
    main()
