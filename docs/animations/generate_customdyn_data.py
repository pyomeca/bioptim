"""
Real solves for ``anim_customdyn.py`` (scene CustomDynamics): a user-defined model that does NOT come from biorbd
(a 1-dof pendulum written directly in CasADi, class ``DampedPendulum`` below, adapted from
bioptim/examples/toy_examples/custom_model/custom_package/my_model.py) is plugged into the OCP of
bioptim/examples/toy_examples/custom_model/main.py (``prepare_ocp``: swing from q = 0 to q = pi in 1 s, N = 30 shooting
intervals, RK4 with 5 steps, minimise int tau^2 dt, |tau| <= 20 N.m).

Two solves, everything else identical:
    - "free"   : damping d = 0            (the example as shipped: its forward_dynamics has ``d = 0``)
    - "damped" : damping d = 1 N.m.s/rad  (the extra viscous term -d * qdot added in the custom dynamics function)
Stored in data/customdyn_pendulum.npz: t, q, qdot, tau (both runs), damping, IPOPT status / iterations / converged /
cost, and the energies computed from the solution (dissipated: trapezoid rule over the N + 1 nodes; work: tau is piecewise constant, so int tau qdot dt = sum tau_k (q_k+1 - q_k)):
    potential_drop (energy released by gravity), dissipated = int d qdot^2 dt   and   work = int tau qdot dt   (work = potential change + dissipated, checked below).

Sanity check: the ``free`` run reproduces the example's own MyModel (same cost), printed at the end.

Usage (env with bioptim on the PATH), from the repo root:
    PYTHONPATH=. python docs/animations/generate_customdyn_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    DynamicsEvaluation,
    Solver,
    SolutionMerge,
    StateDynamics,
    States,
    Controls,
    ConfigureVariables,
)
from casadi import MX, Function, sin, vertcat

from bioptim.examples.toy_examples.custom_model.custom_package import MyModel
from bioptim.examples.toy_examples.custom_model.main import prepare_ocp

OUT = Path(__file__).parent / "data"
T, N = 1.0, 30  # same as custom_model/main.py
DAMPING = 1.0  # N.m.s/rad, the extra term


# ---------------------------------------------------------------------------------------------------------------------
# The custom model: this is the code shown in the video (simplified: comments dropped)
class DampedPendulum(StateDynamics):
    def __init__(self, damping=0.0, **kwargs):
        super().__init__(**kwargs)
        self.damping = damping
        self.com, self.inertia, self.mass = [-0.0005, 0.0688, -0.9542], 0.0391, 1.0
        self.q, self.qdot, self.tau = MX.sym("q", 1), MX.sym("qdot", 1), MX.sym("tau", 1)

    @property
    def name(self):
        return "DampedPendulum"

    @property
    def name_dofs(self):
        return ["rotx"]

    @property
    def state_configuration_functions(self):
        return [
            lambda ocp, nlp: ConfigureVariables.configure_new_variable("q", self.name_dofs, ocp, nlp, as_states=True),
            States.QDOT,
        ]

    @property
    def control_configuration_functions(self):
        return [Controls.TAU]

    @property
    def algebraic_configuration_functions(self):
        return []

    @property
    def extra_configuration_functions(self):
        return []

    def dynamics(self, time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp):
        qddot = self.forward_dynamics()(states[0], states[1], controls[0], [])
        return DynamicsEvaluation(dxdt=vertcat(states[1], qddot), defects=None)

    def forward_dynamics(self):
        L, I, m, g = self.com[2], self.inertia, self.mass, 9.81
        qddot = 1 / (I + m * L**2) * (-self.damping * self.qdot - g * m * L * sin(self.q) + self.tau)
        return Function("forward_dynamics", [self.q, self.qdot, self.tau, MX()], [qddot])


def solve(model):
    ocp = prepare_ocp(model=model, final_time=T, n_shooting=N)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(200)
    sol = ocp.solve(solver=solver)
    st = sol.decision_states(to_merge=SolutionMerge.NODES)
    ct = sol.decision_controls(to_merge=SolutionMerge.NODES)
    return dict(
        q=np.array(st["q"])[0],
        qdot=np.array(st["qdot"])[0],
        tau=np.array(ct["tau"])[0],
        status=int(sol.status),
        iterations=int(sol.iterations),
        cost=float(sol.cost),
    )


if __name__ == "__main__":
    t = np.linspace(0, T, N + 1)
    out = {"t": t, "final_time": T, "n_shooting": N, "damping": DAMPING}
    for tag, d in (("free", 0.0), ("damped", DAMPING)):
        r = solve(DampedPendulum(damping=d))
        for k, v in r.items():
            out[f"{tag}_{k}"] = v
        out[f"{tag}_converged"] = r["status"] == 0
        out[f"{tag}_damping"] = d
        diss = np.trapezoid(d * r["qdot"] ** 2, t)
        work = float(np.sum(r["tau"] * np.diff(r["q"])))  # tau is constant on each interval: int tau qdot dt = tau dq
        out[f"{tag}_dissipated"], out[f"{tag}_work"] = diss, work
        # potential energy V(q) = -g m L cos(q) of the pendulum written in DampedPendulum.forward_dynamics
        L, m, g = -0.9542, 1.0, 9.81
        dV = -g * m * L * (np.cos(r["q"][-1]) - np.cos(r["q"][0]))
        out[f"{tag}_potential_drop"] = -dV  # energy released by gravity between q = 0 and q = pi
        print(
            f"{tag}: d={d} status={r['status']} iterations={r['iterations']} cost={r['cost']:.4f} "
            f"peak|tau|={np.abs(r['tau']).max():.3f} N.m peak|qdot|={np.abs(r['qdot']).max():.3f} "
            f"dissipated={diss:.3f} J work={work:.3f} J dV={dV:.3f} J (work - dissipated - dV = {work - diss - dV:.3f})"
        )
    ref = solve(MyModel())
    print(f"example's own MyModel: status={ref['status']} iterations={ref['iterations']} cost={ref['cost']:.4f}")
    out["example_cost"] = ref["cost"]
    np.savez(OUT / "customdyn_pendulum.npz", **out)
