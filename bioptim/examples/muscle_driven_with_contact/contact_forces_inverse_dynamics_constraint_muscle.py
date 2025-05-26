"""
This example shows how to impose the dynamics through an inverse dynamics defect in collocation.
It also shows how to impose the contact forces as an implicit constraint.
Please note that this formulation does not reach convergence.
"""

import platform

from matplotlib import pyplot as plt
import numpy as np
from casadi import MX, SX, vertcat
from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    ConstraintList,
    ConstraintFcn,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
    Solver,
    SolutionMerge,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsEvaluation,
    DynamicsFunctions,
    ExternalForceSetVariables,
    OdeSolver,
    DefectType,
    InterpolationType,
    PhaseDynamics,
    MultinodeConstraintList,
    MultinodeConstraintFcn,
    ControlType,
    Shooting,
    SolutionIntegrator,
    ContactType,
)


def contact_velocity_all_points(controller):
    contact_velocities = []
    for i_contact in range(2):
        qs = [controller.states["q"].cx_start] + controller.states["q"].cx_intermediates_list
        qdots = [controller.states["qdot"].cx_start] + controller.states["qdot"].cx_intermediates_list
        for i_sn in range(len(qs)):
            contact_axis = controller.model.rigid_contact_axes_index(contact_index=i_contact)
            contact_velocities += [
                controller.model.rigid_contact_velocity(contact_index=i_contact, contact_axis=contact_axis)(
                    qs[i_sn], qdots[i_sn], controller.parameters.cx
                )
            ]
    return vertcat(*contact_velocities)


def contact_velocity_start(controller):
    contact_velocities = []
    q = controller.states["q"].cx_start
    qdot = controller.states["qdot"].cx_start
    for i_contact in range(2):
        contact_axis = controller.model.rigid_contact_axes_index(contact_index=i_contact)
        contact_velocities += [
            controller.model.rigid_contact_velocity(contact_index=i_contact, contact_axis=contact_axis)(
                q, qdot, controller.parameters.cx
            )
        ]
    return vertcat(*contact_velocities)


def prepare_ocp(
    biorbd_model_path,
    phase_time,
    n_shooting,
    defects_type: DefectType,
    contact_type: list[ContactType],
    expand_dynamics=True,
):

    # BioModel
    bio_model = BiorbdModel(biorbd_model_path, contact_type=contact_type)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, index=[1, 2, 3])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, weight=100)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.MUSCLE_DRIVEN,
        with_residual_torque=True,
        expand_dynamics=expand_dynamics,
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3, defects_type=defects_type),
    )

    # Constraints
    constraints = ConstraintList()
    multinode_constraints = MultinodeConstraintList()
    # This constraint is necessary to prevent the contacts from drifting
    if defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
        constraints.add(
            contact_velocity_all_points,
            node=Node.ALL_SHOOTING,
        )
        for i_node in range(n_shooting - 1):
            multinode_constraints.add(
                MultinodeConstraintFcn.ALGEBRAIC_STATES_CONTINUITY,
                nodes_phase=(0, 0),
                nodes=(i_node, i_node + 1),
                key="rigid_contact_forces",
            )
    else:
        constraints.add(
            ConstraintFcn.TRACK_MARKERS_VELOCITY,
            node=Node.START,
            marker_index=0,
        )
        constraints.add(
            ConstraintFcn.TRACK_MARKERS_VELOCITY,
            node=Node.START,
            marker_index=1,
        )

    # Path constraint
    n_q = bio_model.nb_q
    n_qdot = n_q
    pose_at_first_node = [0, 0, -0.75, 0.75]

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

    a_bounds = BoundsList()
    a_init = InitialGuessList()
    if ContactType.RIGID_IMPLICIT in contact_type:
        # Define algebraic states path constraint
        a_bounds.add(
            "rigid_contact_forces",
            min_bound=[-200.0, 0.0, 0.0],
            max_bound=[200.0, 200.0, 200.0],
            interpolation=InterpolationType.CONSTANT,
        )
        a_bounds.add(
            "rigid_contact_forces",
            min_bound=[-200.0, 0.0, 0.0],
            max_bound=[200.0, 200.0, 200.0],
            interpolation=InterpolationType.CONSTANT,
        )

        a_init["rigid_contact_forces"] = [1.0, 1.0, 1.0]

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        phase_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        a_bounds=a_bounds,
        x_init=x_init,
        u_init=u_init,
        a_init=a_init,
        control_type=ControlType.CONSTANT,
        objective_functions=objective_functions,
        constraints=constraints,
        multinode_constraints=multinode_constraints,
    )


def main():
    biorbd_model_path = "models/2segments_4dof_2contacts_1muscle.bioMod"
    t = 0.1
    ns = 100
    defects_type = DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS
    contact_type = [ContactType.RIGID_IMPLICIT]
    ocp = prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        phase_time=t,
        n_shooting=ns,
        defects_type=defects_type,
        contact_type=contact_type,
    )
    # ocp.add_plot_penalty()

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
    q = states["q"]
    qdot = states["qdot"]
    tau_residual = controls["tau"]
    mus = controls["muscles"]

    if DefectType.TAU_EQUALS_INVERSE_DYNAMICS in defects_type:
        contact_forces = algebraic_states["rigid_contact_forces"]

        # --- Get contact position --- #
        contact_positions = np.zeros((2, 3, time.shape[0]))
        for i_node in range(time.shape[0]):
            for i_contact in range(2):
                contact_positions[i_contact, :, i_node] = np.reshape(
                    nlp.model.rigid_contact_position(i_contact)(q[:, i_node], []), (3,)
                )

        # --- Plots --- #
        fig, axs = plt.subplots(2, 1, figsize=(10, 7))
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

        names_contact_forces = ocp.nlp[0].model.rigid_contact_names
        for i_ax in range(len(names_contact_forces)):
            axs[1].plot(time, contact_forces[i_ax, :], ".-", label=f"{names_contact_forces[i_ax]}")
        axs[1].legend()
        axs[1].grid()
        axs[1].set_title("Contact forces [N]")
        plt.savefig("test.png")
        plt.show()

    # --- Plot the reintegration -- #
    sol_integrated = sol.integrate(
        shooting_type=Shooting.SINGLE,
        integrator=SolutionIntegrator.SCIPY_DOP853,
        to_merge=SolutionMerge.NODES,
        return_time=False,
    )
    time_integrated = np.linspace(0, t, sol_integrated["q"].shape[1])
    q_integrated, qdot_integrated = sol_integrated["q"], sol_integrated["qdot"]

    nb_q = nlp.model.nb_q
    fig, axs = plt.subplots(nb_q, 1, figsize=(10, 7))
    for i_dof in range(nb_q):
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
            time_integrated, q_integrated[i_dof, :], ".", linestyle="none", color="tab:red", label="Reintegration - q"
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
            time_integrated,
            qdot_integrated[i_dof, :],
            ".",
            linestyle="none",
            color="tab:blue",
            label="Reintegration - qdot",
        )
        axs[i_dof].set_title(f"{ocp.nlp[0].model.name_dof[i_dof]}")
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"reintegration_{defects_type.value}.png")
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
