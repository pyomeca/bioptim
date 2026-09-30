"""
Discrete mechanics with bioptim's variational model (VariationalTorqueBiorbdModel), on a 1 m / 1 kg simple pendulum
(docs/animations/models/variational_pendulum.bioMod, q = 0 hanging down).

Part 1, free motion (tau = 0), same time step for every scheme, released at rest from q0 = 90 deg:
  * variational: the model's OWN discrete Euler-Lagrange equations (`discrete_euler_lagrange_equations`, and
    `compute_initial_states` for the first step) are solved for q_{k+1} with a casadi root finder. Nothing is
    integrated by a Runge-Kutta scheme; the velocity is deduced from the discrete momentum
    p_k = D2 Ld(q_{k-1}, q_k) through the continuous Legendre transform qdot_k = M(q_k)^-1 p_k.
  * explicit Euler (RK1) and RK4 on qddot = model.forward_dynamics(q, qdot, tau=0), one step per dt.
  * total energy E = T + V is computed for every scheme with the model's own Lagrangian: E = L(q,qdot) - 2 L(q,0).
Part 2, a real VariationalOptimalControlProgram (swing-up, same structure as
examples/toy_examples/discrete_mechanics_and_optimal_control/example_variational_integrator_pendulum.py) solved by IPOPT.

Run from the repo root:  PYTHONPATH=. python docs/animations/generate_variational_data.py
"""

from pathlib import Path

import numpy as np
from casadi import DM, MX, Function, jacobian, rootfinder, transpose, vertcat
from bioptim import (
    BoundsList,
    ControlType,
    InitialGuessList,
    InterpolationType,
    Objective,
    ObjectiveFcn,
    Solver,
    SolutionMerge,
    VariationalOptimalControlProgram,
    VariationalTorqueBiorbdModel,
)

OUT = Path(__file__).parent / "data"
MODEL = str(Path(__file__).parent / "models" / "variational_pendulum.bioMod")

DT = 0.1  # time step shared by the three schemes
T_END = 600.0
Q0 = np.pi / 2  # released at rest


def free_motion():
    n = int(round(T_END / DT))
    m = VariationalTorqueBiorbdModel(MODEL, control_type=ControlType.LINEAR_CONTINUOUS)
    lag = m.lagrangian()
    fd = m.forward_dynamics()
    mass = m.mass_matrix()
    zero = DM.zeros(1, 1)
    p0 = DM.zeros(m.parameters.shape) if hasattr(m.parameters, "shape") else DM.zeros(0, 1)

    def energy(q, qd):
        return float(lag(q, qd) - 2 * lag(q, 0))

    # ------------------------------------------------ variational: discrete Euler-Lagrange equations
    qa, qb, qc = MX.sym("qa", 1), MX.sym("qb", 1), MX.sym("qc", 1)
    dt = MX.sym("dt")
    del_res = m.discrete_euler_lagrange_equations(dt, qa, qb, qc, MX.zeros(1), MX.zeros(1), MX.zeros(1))
    step = rootfinder("del", "newton", Function("f", [qc, qa, qb, dt], [del_res]))
    qd0 = MX.sym("qd0", 1)
    first_res = m.compute_initial_states(dt, qa, qd0, qc, MX.zeros(1), MX.zeros(1))
    first = rootfinder("first", "newton", Function("g", [qc, qa, qd0, dt], [first_res]))
    mom = Function(
        "mom", [qa, qb, dt], [transpose(jacobian(m.discrete_lagrangian(qa, qb, dt), qb))]
    )  # D2 Ld(q_{k-1}, q_k)

    q = np.zeros(n + 1)
    qd = np.zeros(n + 1)
    q[0] = Q0
    q[1] = float(first(q[0], q[0], 0.0, DT))
    for k in range(1, n):
        q[k + 1] = float(step(q[k], q[k - 1], q[k], DT))
    for k in range(1, n + 1):
        p = float(mom(q[k - 1], q[k], DT))
        qd[k] = p / float(mass(q[k], p0))
    e_var = np.array([energy(q[k], qd[k]) for k in range(n + 1)])
    res_max = max(
        abs(float(step(q[k + 1], q[k - 1], q[k], DT) - q[k + 1])) for k in range(1, n)
    )  # sanity (0 by design)

    # ------------------------------------------------ explicit RK1 / RK4
    def f(x):
        return np.array([x[1], float(fd(x[0], x[1], zero, DM.zeros(fd.size1_in(3), 1), p0))])

    def rk1(x):
        return x + DT * f(x)

    def rk4(x):
        k1 = f(x)
        k2 = f(x + DT / 2 * k1)
        k3 = f(x + DT / 2 * k2)
        k4 = f(x + DT * k3)
        return x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    out = dict(dt=DT, t_end=T_END, q0=Q0, t=np.arange(n + 1) * DT, q_var=q, qdot_var=qd, e_var=e_var)
    for name, scheme in (("rk1", rk1), ("rk4", rk4)):
        x = np.array([Q0, 0.0])
        xs = [x]
        for _ in range(n):
            x = scheme(x)
            xs.append(x)
        xs = np.array(xs)
        out["q_" + name] = xs[:, 0]
        out["qdot_" + name] = xs[:, 1]
        out["e_" + name] = np.array([energy(a, b) for a, b in xs])
    e0 = float(out["e_rk4"][0])
    out["e0"] = e0
    print(f"E0 = {e0:.4f} J (analytic m g l (1-cos q0) + ... = {9.81 * (1 - np.cos(Q0)):.4f} above the hanging rest)")
    for name in ("var", "rk1", "rk4"):
        e = out["e_" + name]
        print(f"{name}: E(0)={e[0]:.4f}  E(end)={e[-1]:.4f}  max|E-E0|={np.abs(e - e0).max():.4f}  ")
    print("variational root-finder residual", res_max)
    return out


def swing_up():
    n_shooting, final_time = 50, 2.0
    bio_model = VariationalTorqueBiorbdModel(MODEL)
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0
    x_bounds["q"][:, -1] = np.pi
    x_init = InitialGuessList()
    x_init.add("q", initial_guess=[np.linspace(0, np.pi, n_shooting + 1)], interpolation=InterpolationType.EACH_FRAME)
    u_bounds = BoundsList()
    u_bounds["tau"] = [-TAU_MAX], [TAU_MAX]
    qdot_bounds = BoundsList()
    qdot_bounds.add("qdot_start", min_bound=[0], max_bound=[0], interpolation=InterpolationType.CONSTANT)
    qdot_bounds.add("qdot_end", min_bound=[0], max_bound=[0], interpolation=InterpolationType.CONSTANT)
    ocp = VariationalOptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        q_bounds=x_bounds,
        u_bounds=u_bounds,
        qdot_bounds=qdot_bounds,
        q_init=x_init,
        objective_functions=objective_functions,
        use_sx=True,
    )
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver)
    q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"][0]
    tau = sol.decision_controls(to_merge=SolutionMerge.NODES)["tau"][0]
    print("swing-up status", sol.status, "iterations", sol.iterations, "cost", float(sol.cost))
    print("q range", q.min(), q.max(), "tau range", np.nanmin(tau), np.nanmax(tau), "tau shape", tau.shape)
    return dict(
        su_q=q,
        su_tau=tau,
        su_t=np.linspace(0, final_time, n_shooting + 1),
        su_status=sol.status,
        su_iterations=sol.iterations,
        su_cost=float(sol.cost),
        su_tau_max=TAU_MAX,
        su_n=n_shooting,
        su_final_time=final_time,
        su_qdot_start=float(np.squeeze(sol.parameters["qdot_start"])),
        su_qdot_end=float(np.squeeze(sol.parameters["qdot_end"])),
    )


TAU_MAX = 25.0

if __name__ == "__main__":
    out = free_motion()
    out.update(swing_up())
    np.savez(OUT / "variational_pendulum.npz", **out)
