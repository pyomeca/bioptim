"""
Which dynamics can a Bioptim model have?  The SAME swing task solved with four dynamics (REAL bioptim / IPOPT solves), plus
the state / control variables of the other dynamics read from REAL (unsolved) optimal control problems.
Stored in ``data/dynamics_compare.npz`` and ``data/dynamics_names.npz``.

Task (models/dynamics_acrobot.bioMod = bioptim/examples/models/double_pendulum.bioMod without meshes, plus torque
actuators of 20 N.m on both joints, needed by TorqueActivationBiorbdModel): hanging at rest, the FIRST joint (the root,
q[0]) is never actuated; the second joint (the elbow, q[1], range +-pi/2) is the only actuator. Reach q[0] = 1 rad at rest
(qdot = 0) at the final node; the elbow angle is free at the end. N = 30 intervals, T = 4 s, RK4 (5 steps), multiple
shooting, IPOPT, default initial guess, NO warm start, NO continuation.

Four dynamics, each minimising the integral of the squared value of ITS OWN control (weight 1):
    tag  model                           states              control         bounds on the control
    tau  TorqueBiorbdModel               q, qdot             tau (N.m)       +-20 (tau[0] = 0)
    act  TorqueActivationBiorbdModel     q, qdot             tau (activation) +-1 (tau[0] = 0), torque = a * Tmax(20)
    der  TorqueDerivativeBiorbdModel     q, qdot, tau        taudot (N.m/s)  +-500 (taudot[0] = 0); tau state +-20, 0 at t = 0
    acc  JointAccelerationBiorbdModel    q, qdot             qddot_joints (rad/s^2)  +-100 (the root is passive: 1 control)
The four costs have DIFFERENT units and are NOT comparable. What can be compared is the physical joint torque tau_2
of the motion: tau, act.tau * Tmax, the state tau, and the inverse-dynamics torque of the accelerations (acc).
``tau2_effort`` is the integral of tau_2^2 dt with tau_2 taken at the start of each interval (the controls are constant
per interval) and ``tau2_peak`` the largest |tau_2| over the same values. The trapezoid of the states is not used, so
the two quantities are approximations of the same accuracy for the four dynamics.

Also stored (``dynamics_names.npz``): the names and sizes of the states / controls of REAL OCPs built (not solved) with
MusclesBiorbdModel, MusclesWithExcitationsBiorbdModel (arm26), TorqueFreeFloatingBaseBiorbdModel and MultiTorqueBiorbdModel.
Holonomic, variational and stochastic variables are read from the classes (bioptim/dynamics/state_space_dynamics/).

Usage, from the repo root:  PYTHONPATH=. python docs/animations/generate_dynamics_data.py
"""

from pathlib import Path

import numpy as np
from bioptim import (
    BoundsList,
    DynamicsOptions,
    JointAccelerationBiorbdModel,
    MultiTorqueBiorbdModel,
    MusclesBiorbdModel,
    MusclesWithExcitationsBiorbdModel,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    TorqueActivationBiorbdModel,
    TorqueBiorbdModel,
    TorqueDerivativeBiorbdModel,
    TorqueFreeFloatingBaseBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils

HERE = Path(__file__).parent
MODEL = str(HERE / "models" / "dynamics_acrobot.bioMod")
OUT = HERE / "data"
N, T = 30, 4.0
Q_END = 1.0  # final angle of the first (passive) joint, at rest
TAU_MAX = 20.0  # N.m, Tmax of the actuators of the bioMod
TAG = ("tau", "act", "der", "acc")


def prepare_ocp(tag):
    if tag == "tau":
        bio_model = TorqueBiorbdModel(MODEL)
        key = "tau"
    elif tag == "act":
        bio_model = TorqueActivationBiorbdModel(MODEL)
        key = "tau"
    elif tag == "der":
        bio_model = TorqueDerivativeBiorbdModel(MODEL)
        key = "taudot"
    else:
        bio_model = JointAccelerationBiorbdModel(MODEL)
        key = "qddot_joints"

    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key=key, weight=1)

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"][:, 0] = 0
    x_bounds["q"][0, -1] = Q_END
    x_bounds["qdot"][:, -1] = 0

    u_bounds = BoundsList()
    if tag == "tau":
        u_bounds["tau"] = [-TAU_MAX, -TAU_MAX], [TAU_MAX, TAU_MAX]
        u_bounds["tau"][0, :] = 0
    elif tag == "act":
        u_bounds["tau"] = [-1, -1], [1, 1]
        u_bounds["tau"][0, :] = 0
    elif tag == "der":
        x_bounds["tau"] = [-TAU_MAX, -TAU_MAX], [TAU_MAX, TAU_MAX]
        x_bounds["tau"][0, :] = 0
        x_bounds["tau"][:, 0] = 0
        u_bounds["taudot"] = [-500, -500], [500, 500]
        u_bounds["taudot"][0, :] = 0
    else:
        u_bounds["qddot_joints"] = [-100], [100]

    ocp = OptimalControlProgram(
        bio_model,
        N,
        T,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(n_integration_steps=5)),
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objectives,
        use_sx=True,
    )
    return ocp, bio_model, key


def joint_torque(tag, bio_model, q, qdot, u, state_tau):
    """Physical torque of joint 2 at the START of each interval k = 0..N-1 (the controls are constant per interval)."""
    tau2 = np.zeros(N)
    for k in range(N):
        if tag == "tau":
            tau2[k] = u[1, k]
        elif tag == "act":
            tau2[k] = float(bio_model.torque()(u[:, k], q[:, k], qdot[:, k], [])[1])
        elif tag == "der":
            tau2[k] = state_tau[1, k]
        else:
            qddot_root = bio_model.forward_dynamics_free_floating_base()(q[:, k], qdot[:, k], u[:, k], [])
            qddot = bio_model.reorder_qddot_root_joints(qddot_root, u[:, k])
            tau = bio_model.inverse_dynamics()(q[:, k], qdot[:, k], qddot, [], [])
            tau2[k] = float(tau[1])
            assert abs(float(tau[0])) < 1e-6, "the root must stay passive"
    return tau2


def solve(tag):
    ocp, bio_model, key = prepare_ocp(tag)
    nlp = ocp.nlp[0]
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(1000)
    sol = ocp.solve(solver)
    st, co = sol.decision_states(), sol.decision_controls()
    q = np.array([st["q"][k][:, 0] for k in range(N + 1)]).T
    qdot = np.array([st["qdot"][k][:, 0] for k in range(N + 1)]).T
    u = np.array([co[key][k][:, 0] for k in range(N)]).T  # (n_u, N), constant per interval
    state_tau = np.array([st["tau"][k][:, 0] for k in range(N + 1)]).T if tag == "der" else None
    tau2 = joint_torque(tag, bio_model, q, qdot, u, state_tau)
    dt = T / N
    res = {
        f"{tag}_t": np.linspace(0, T, N + 1),
        f"{tag}_q": q,
        f"{tag}_qdot": qdot,
        f"{tag}_u": u,
        f"{tag}_u_key": key,
        f"{tag}_tau2": tau2,
        f"{tag}_tau2_peak": float(np.abs(tau2).max()),
        f"{tag}_tau2_effort": float(np.sum(tau2**2) * dt),
        f"{tag}_cost": float(sol.cost),
        f"{tag}_status": int(sol.status),
        f"{tag}_iterations": int(sol.iterations),
        f"{tag}_converged": bool(sol.status == 0),
        f"{tag}_n_vector": int(np.array(sol.vector).size),
        f"{tag}_n_states": int(nlp.states.shape),
        f"{tag}_n_controls": int(nlp.controls.shape),
        f"{tag}_state_names": np.array(list(nlp.states.keys())),
        f"{tag}_control_names": np.array(list(nlp.controls.keys())),
        f"{tag}_q_end": float(q[0, -1]),
    }
    print(
        f"{tag}: status={sol.status} it={sol.iterations} cost={float(sol.cost):.4f} vector={res[f'{tag}_n_vector']} "
        f"states={res[f'{tag}_state_names']} ({res[f'{tag}_n_states']}) controls={res[f'{tag}_control_names']} "
        f"({res[f'{tag}_n_controls']}) key={key} max|u|={np.abs(u).max():.3f} tau2 peak={res[f'{tag}_tau2_peak']:.2f} "
        f"effort={res[f'{tag}_tau2_effort']:.2f} q0_end={q[0, -1]:.3f} q1 range=[{q[1].min():.2f}, {q[1].max():.2f}]"
    )
    return res


def names_of(label, bio_model):
    """Build (not solve) a small OCP with the model's default dynamics and read what is a state and what is a control."""
    ocp = OptimalControlProgram(
        bio_model,
        5,
        1.0,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4()),
        objective_functions=ObjectiveList(),
        use_sx=True,
    )
    nlp = ocp.nlp[0]
    states = {k: len(nlp.states[k]) for k in nlp.states.keys()}
    controls = {k: len(nlp.controls[k]) for k in nlp.controls.keys()}
    print(f"{label}: states {states} controls {controls}")
    return {
        f"{label}_states": np.array([f"{k}:{n}" for k, n in states.items()]),
        f"{label}_controls": np.array([f"{k}:{n}" for k, n in controls.items()]),
    }


def main():
    out = {"N": N, "T": T, "q_end": Q_END, "tau_max": TAU_MAX}
    for tag in TAG:
        out.update(solve(tag))
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "dynamics_compare.npz", **out)

    arm = ExampleUtils.folder + "/models/arm26.bioMod"
    pend = ExampleUtils.folder + "/models/pendulum.bioMod"
    names = {}
    names.update(names_of("muscles", MusclesBiorbdModel(arm)))
    names.update(names_of("muscles_resid", MusclesBiorbdModel(arm, with_residual_torque=True)))
    names.update(names_of("excit", MusclesWithExcitationsBiorbdModel(arm)))
    names.update(names_of("floating", TorqueFreeFloatingBaseBiorbdModel(MODEL)))
    names.update(names_of("multi", MultiTorqueBiorbdModel((pend, pend))))
    np.savez(OUT / "dynamics_names.npz", **names)


if __name__ == "__main__":
    main()
