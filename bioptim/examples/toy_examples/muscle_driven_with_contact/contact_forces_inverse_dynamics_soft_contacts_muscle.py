"""
This example shows how to impose the dynamics through an inverse dynamics defect in collocation.
It also shows how to impose the soft contact forces as an implicit constraint.
"""

from bioptim import (
    MusclesBiorbdModel,
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsOptions,
    BoundsList,
    InitialGuessList,
    Solver,
    SolutionMerge,
    DynamicsFunctions,
    OdeSolver,
    InterpolationType,
    PhaseDynamics,
    MultinodeConstraintList,
    MultinodeConstraintFcn,
    ControlType,
    ContactType,
    DefectType,
)
from bioptim.examples.utils import ExampleUtils
from casadi import MX
from matplotlib import pyplot as plt
import numpy as np


def prepare_ocp(biorbd_model_path, phase_time, n_shooting, expand_dynamics=True):

    # BioModel
    bio_model = MusclesBiorbdModel(
        biorbd_model_path, with_residual_torque=True, contact_types=[ContactType.SOFT_IMPLICIT]
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, index=[1, 2, 3])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, weight=100)

    # Dynamics
    dynamics = DynamicsOptions(
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3, defects_type=DefectType.TAU_EQUALS_INVERSE_DYNAMICS),
    )

    multinode_constraints = MultinodeConstraintList()
    for i_node in range(n_shooting - 1):
        multinode_constraints.add(
            MultinodeConstraintFcn.ALGEBRAIC_STATES_CONTINUITY,
            nodes_phase=(0, 0),
            nodes=(i_node, i_node + 1),
            key="soft_contact_forces",
        )

    # Path constraint
    n_q = bio_model.nb_q
    n_qdot = n_q
    pose_at_first_node = [0, 1, -0.75, 0.75]

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = pose_at_first_node
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init["q"] = pose_at_first_node
    x_init["qdot"] = np.zeros((n_qdot,))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [-200.0] * bio_model.nb_tau, [200.0] * bio_model.nb_tau
    u_bounds["muscles"] = [0.0] * bio_model.nb_muscles, [1.0] * bio_model.nb_muscles

    u_init = InitialGuessList()
    u_init["tau"] = [1.0] * bio_model.nb_tau
    u_init["muscles"] = [0.5] * bio_model.nb_muscles

    # # Define algebraic states path constraint
    a_bounds = BoundsList()
    a_bounds.add(
        "soft_contact_forces",
        min_bound=[-200, -200, -200, -10, -200, 0, -200, -200, -200, -10, -200, 0],
        max_bound=[200, 200, 200, 10, 200, 200, 200, 200, 200, 10, 200, 200],
        interpolation=InterpolationType.CONSTANT,
    )

    a_init = InitialGuessList()
    a_init["soft_contact_forces"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        phase_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        a_bounds=a_bounds,
        x_init=x_init,
        u_init=u_init,
        a_init=a_init,
        control_type=ControlType.LINEAR_CONTINUOUS,
        objective_functions=objective_functions,
        multinode_constraints=multinode_constraints,
    )


def main():

    # This example does not converge, but it is a good example of how to set up the problem
    # And the dynamics seems fine (inf_pr = 5.66e-07) when restoration failed

    biorbd_model_path = ExampleUtils.folder + "/models/2segments_4dof_2soft_contacts_1muscle.bioMod"
    t = 1
    ns = 100
    ocp = prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        phase_time=t,
        n_shooting=ns,
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(10000)
    sol = ocp.solve(solver)
    nlp = ocp.nlp[0]
    # sol.graphs()

    time = np.reshape(sol.decision_time(to_merge=SolutionMerge.NODES), (-1,))
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    algebraic_states = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)
    q, qdot, tau_residual, mus, contact_forces = (
        states["q"],
        states["qdot"],
        controls["tau"],
        controls["muscles"],
        algebraic_states["soft_contact_forces"],
    )

    # --- Get contact position --- #
    contact_positions = np.zeros((2, 3, time.shape[0]))
    for i_node in range(time.shape[0]):
        for i_contact in range(2):
            contact_positions[i_contact, :, i_node] = np.reshape(nlp.model.marker(i_contact)(q[:, i_node], []), (3,))

    # --- Plots --- #
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    names_contact_points = ["Seg1_contact1", "Seg1_contact2"]
    colors = ["tab:red", "tab:blue"]
    for i_contact in range(len(names_contact_points)):
        axs[0].plot(
            time,
            contact_positions[i_contact, 0, :],
            ".-",
            color=colors[i_contact],
            label=f"{names_contact_points[i_contact]} - x",
        )
        axs[0].plot(
            time,
            contact_positions[i_contact, 1, :],
            "--",
            color=colors[i_contact],
            label=f"{names_contact_points[i_contact]} - y",
        )
        axs[0].plot(
            time,
            contact_positions[i_contact, 2, :],
            ":",
            color=colors[i_contact],
            label=f"{names_contact_points[i_contact]} - z",
        )
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title("Contact position [m]")

    names_contact_forces = ocp.nlp[0].model.soft_contact_names
    axs[1].plot(time, contact_forces[0, :], ".-", label=f"{names_contact_forces[0]} - Mx", color=colors[0], alpha=0.5)
    axs[1].plot(time, contact_forces[1, :], "--", label=f"{names_contact_forces[0]} - My", color=colors[0], alpha=0.5)
    axs[1].plot(time, contact_forces[2, :], ":", label=f"{names_contact_forces[0]} - Mz", color=colors[0], alpha=0.5)
    axs[1].plot(time, contact_forces[3, :], ".-", label=f"{names_contact_forces[0]} - Fx", color=colors[0])
    axs[1].plot(time, contact_forces[4, :], "--", label=f"{names_contact_forces[0]} - Fy", color=colors[0])
    axs[1].plot(time, contact_forces[5, :], ":", label=f"{names_contact_forces[0]} - Fz", color=colors[0])
    axs[1].plot(time, contact_forces[6, :], ".-", label=f"{names_contact_forces[1]} - Mx", color=colors[1], alpha=0.5)
    axs[1].plot(time, contact_forces[7, :], "--", label=f"{names_contact_forces[1]} - My", color=colors[1], alpha=0.5)
    axs[1].plot(time, contact_forces[8, :], ":", label=f"{names_contact_forces[1]} - Mz", color=colors[1], alpha=0.5)
    axs[1].plot(time, contact_forces[9, :], ".-", label=f"{names_contact_forces[1]} - Fx", color=colors[1])
    axs[1].plot(time, contact_forces[10, :], "--", label=f"{names_contact_forces[1]} - Fy", color=colors[1])
    axs[1].plot(time, contact_forces[11, :], ":", label=f"{names_contact_forces[1]} - Fz", color=colors[1])
    axs[1].legend()
    axs[1].grid()
    axs[1].set_title("Contact forces [N]")
    plt.savefig("contacts.png")
    plt.show()

    # --- TODO: REMOVE --- #
    def integrate(time, q, qdot, tau, muscle, forward_dynamics_func):

        def get_u(t0, t, dt, u0, u1):
            dt_norm = (t - t0) / dt
            return u0 + (u1 - u0) * dt_norm

        n_shooting = time.shape[0] - 1
        dt = time[1] - time[0]
        h = dt / 5
        q_integrated = np.zeros((4, n_shooting + 1))
        q_integrated[:, 0] = q[:, 0]
        qdot_integrated = np.zeros((4, n_shooting + 1))
        qdot_integrated[:, 0] = qdot[:, 0]
        for i_shooting in range(n_shooting):
            q_this_time = q_integrated[:, i_shooting]
            qdot_this_time = qdot_integrated[:, i_shooting]
            t0 = i_shooting * dt
            for i_step in range(5):
                t = t0 + i_step * h
                q_dot1 = qdot_this_time[:]
                qdot_dot1 = forward_dynamics_func(
                    cas.vertcat(q_this_time, qdot_this_time),
                    cas.vertcat(
                        get_u(t0, t, dt, tau[:, i_shooting], tau[:, i_shooting + 1]),
                        get_u(t0, t, dt, muscle[:, i_shooting], muscle[:, i_shooting + 1]),
                    ),
                )
                q_dot2 = qdot_this_time[:] + h / 2 * qdot_dot1
                qdot_dot2 = forward_dynamics_func(
                    cas.vertcat(q_this_time + h / 2 * q_dot1, qdot_this_time + h / 2 * qdot_dot1),
                    cas.vertcat(
                        get_u(t0, t + h / 2, dt, tau[:, i_shooting], tau[:, i_shooting + 1]),
                        get_u(t0, t + h / 2, dt, muscle[:, i_shooting], muscle[:, i_shooting + 1]),
                    ),
                )
                q_dot3 = qdot_this_time[:] + h / 2 * qdot_dot2
                qdot_dot3 = forward_dynamics_func(
                    cas.vertcat(q_this_time + h / 2 * q_dot2, qdot_this_time + h / 2 * qdot_dot2),
                    cas.vertcat(
                        get_u(t0, t + h / 2, dt, tau[:, i_shooting], tau[:, i_shooting + 1]),
                        get_u(t0, t + h / 2, dt, muscle[:, i_shooting], muscle[:, i_shooting + 1]),
                    ),
                )
                q_dot4 = qdot_this_time[:] + h * qdot_dot3
                qdot_dot4 = forward_dynamics_func(
                    cas.vertcat(q_this_time + h * q_dot3, qdot_this_time + h * qdot_dot3),
                    cas.vertcat(
                        get_u(t0, t + h, dt, tau[:, i_shooting], tau[:, i_shooting + 1]),
                        get_u(t0, t + h, dt, muscle[:, i_shooting], muscle[:, i_shooting + 1]),
                    ),
                )
                q_this_time = q_this_time + h / 6 * (q_dot1 + 2 * q_dot2 + 2 * q_dot3 + q_dot4)
                qdot_this_time = qdot_this_time + h / 6 * (qdot_dot1 + 2 * qdot_dot2 + 2 * qdot_dot3 + qdot_dot4)
            q_integrated[:, i_shooting + 1] = np.reshape(q_this_time, (4,))
            qdot_integrated[:, i_shooting + 1] = np.reshape(qdot_this_time[:, 0], (4,))
        return q_integrated, qdot_integrated

    import casadi as cas

    nlp = ocp.nlp[0]
    # Variables
    q_sym = nlp.states["q"].cx
    qdot_sym = nlp.states["qdot"].cx
    residual_tau_sym = nlp.controls["tau"].cx
    mus_activations_sym = nlp.controls["muscles"].cx
    soft_contact_forces_sym = nlp.algebraic_states["soft_contact_forces"].cx

    # Compute joint torques
    muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q_sym, qdot_sym, mus_activations_sym)
    tau = muscles_tau + residual_tau_sym

    soft_contact_forces_computed = nlp.model.soft_contact_forces().expand()(q_sym, qdot_sym, nlp.parameters.cx)

    # Map to external forces
    external_forces = MX.zeros(9 * 2)
    external_forces[0:6] = soft_contact_forces_computed[0:6]
    external_forces[9:15] = soft_contact_forces_computed[6:12]

    ddq = nlp.model.forward_dynamics(with_contact=False)(q_sym, qdot_sym, tau, external_forces, [])

    forward_dynamics_func = cas.Function("forward_dyn", [nlp.states.cx, nlp.controls.cx], [ddq])

    q_integrated, qdot_integrated = integrate(
        time[0::4], q[:, 0::4], qdot[:, 0::4], tau_residual[:, 0::2], mus[:, 0::2], forward_dynamics_func
    )

    # # --- Plot the reintegration to confirm dynamics consistency --- #
    # sol_integrated = sol.integrate(shooting_type=Shooting.SINGLE,
    #                                 integrator=SolutionIntegrator.SCIPY_DOP853,
    #                                 to_merge=SolutionMerge.NODES,
    #                                 return_time=False,
    #                                )
    # q_integrated, qdot_integrated = sol_integrated["q"], sol_integrated["qdot"]

    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    for i_dof in range(4):
        axs[i_dof].plot(
            time,
            q[i_dof, :],
            marker="o",
            linestyle="none",
            fillstyle="none",
            color="tab:red",
            label="Optimal solution - q",
        )
        axs[i_dof].plot(
            time[0::4], q_integrated[i_dof, :], ".", linestyle="none", color="tab:red", label="Reintegration - q"
        )
        axs[i_dof].plot(
            time,
            qdot[i_dof, :],
            marker="o",
            linestyle="none",
            fillstyle="none",
            color="tab:blue",
            label="Optimal solution - qdot",
        )
        axs[i_dof].plot(
            time[0::4], qdot_integrated[i_dof, :], ".", linestyle="none", color="tab:blue", label="Reintegration - qdot"
        )
        axs[i_dof].set_title(f"{ocp.nlp[0].model.name_dofs[i_dof]}")
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("reintegration.png")
    plt.show()

    # --- Show results --- #
    viewer = "pyorerun"
    if viewer == "pyorerun":
        from pyorerun import BiorbdModel, PhaseRerun

        # Model
        model = BiorbdModel(biorbd_model_path)
        model.options.transparent_mesh = False
        model.options.show_gravity = True
        model.options.show_floor = True

        # Visualization
        viz = PhaseRerun(time)
        viz.add_animated_model(model, q)
        viz.rerun_by_frame("Optimal solution")
    else:
        sol.animate()


if __name__ == "__main__":
    main()
