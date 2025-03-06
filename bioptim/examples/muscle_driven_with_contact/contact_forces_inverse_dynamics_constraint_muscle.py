"""
This example shows how to impose the dynamics through an inverse dynamics defect in collocation.
It also shows how to impose the contact forces as an implicit constraints.
"""

import platform

from matplotlib import pyplot as plt
import numpy as np
from casadi import MX, SX, horzcat, vertcat
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
    BiMappingList,
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
    MultinodeConstraintList,
    MultinodeConstraintFcn,
)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None):
    # Usual variables
    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False, as_states_dot=True)
    ConfigureProblem.configure_qddot(ocp, nlp, as_states=False, as_controls=False, as_states_dot=True)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)  # Residual torques
    ConfigureProblem.configure_muscles(ocp, nlp, as_states=False, as_controls=True)  # Muscle activation

    # Implicit variables
    ConfigureProblem.configure_rigid_contact_forces(
        ocp, nlp, as_states=False, as_algebraic_states=True, as_controls=False
    )

    # Dynamics
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics)


def custom_dynamics(
    time: MX | SX,
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    algebraic_states: MX | SX,
    numerical_timeseries: MX | SX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    The defects are only evaluated at the collocation nodes, but not at the first node.
    """

    # Variables
    q = nlp.get_var_from_states_or_controls("q", states, controls)
    qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
    residual_tau = nlp.get_var_from_states_or_controls("tau", states, controls)
    mus_activations = nlp.get_var_from_states_or_controls("muscles", states, controls)

    # Get external forces from algebraic states
    rigid_contact_forces = nlp.get_external_forces(
        "rigid_contact_forces", states, controls, algebraic_states, numerical_timeseries
    )
    # Map to external forces
    external_forces = nlp.model.map_rigid_contact_forces_to_global_forces(rigid_contact_forces, q, parameters)

    # Compute joint torques
    muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations)
    tau = muscles_tau + residual_tau

    # Defects
    slope_q = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
    slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.scaled.cx)
    tau_id = DynamicsFunctions.inverse_dynamics(
        nlp, q, slope_q, slope_qdot, with_contact=False, external_forces=external_forces
    )
    defects = vertcat(qdot - slope_q, tau - tau_id)

    return DynamicsEvaluation(dxdt=None, defects=defects)


def custom_first_defect(controller):
    """
    The defect at the first node.
    """

    # Variables
    q = controller.states["q"].cx_start
    qdot = controller.states["qdot"].cx_start
    residual_tau = controller.controls["tau"].cx_start
    mus_activations = controller.controls["muscles"].cx_start

    # Get external forces from algebraic states
    rigid_contact_forces = controller.get_nlp.get_external_forces(
        "rigid_contact_forces",
        controller.states.cx,
        controller.controls.cx,
        controller.algebraic_states.cx,
        controller.numerical_timeseries.cx,
    )
    # Map to external forces
    external_forces = controller.model.map_rigid_contact_forces_to_global_forces(
        rigid_contact_forces, q, controller.parameters.cx
    )

    # Compute joint torques
    muscles_tau = DynamicsFunctions.compute_tau_from_muscle(controller.get_nlp, q, qdot, mus_activations)
    tau = muscles_tau + residual_tau

    # Defects
    slope_q = controller.states_dot["qdot"].cx_start
    slope_qdot = controller.states_dot["qddot"].cx_start
    tau_id = DynamicsFunctions.inverse_dynamics(
        controller.get_nlp, q, slope_q, slope_qdot, with_contact=False, external_forces=external_forces
    )
    defects = vertcat(qdot - slope_q, tau - tau_id)

    return defects


def contact_velocity(controller):
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


def prepare_ocp(biorbd_model_path, phase_time, n_shooting, expand_dynamics=True):

    # Indicate to the model creator that there will be two rigid contacts in the form of optimization variables
    external_force_set = ExternalForceSetVariables()
    external_force_set.add("Seg1", use_point_of_application=True)
    external_force_set.add("Seg1", use_point_of_application=True)

    # BioModel
    bio_model = BiorbdModel(biorbd_model_path, external_force_set=external_force_set)
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", bimapping=None, to_second=[None, None, None, 0], to_first=[3])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_CONTROL, key="tau", weight=1)
    # objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=10, index=, target=, node=Node.ALL_SHOOTING)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        custom_configure,
        dynamic_function=custom_dynamics,
        expand_dynamics=expand_dynamics,
    )

    # Constraints
    constraints = ConstraintList()
    # This constraint is necessary to prevent the contacts from drifting
    constraints.add(
        contact_velocity,
        node=Node.ALL_SHOOTING,
    )
    constraints.add(
        custom_first_defect,
        node=Node.ALL_SHOOTING,
    )
    multinode_constraints = MultinodeConstraintList()
    for i_node in range(n_shooting):
        multinode_constraints.add(
            MultinodeConstraintFcn.ALGEBRAIC_STATES_CONTINUITY,
            nodes_phase=(0, 0),
            nodes=(i_node, i_node + 1),
            sub_nodes=(-1, 0),
            key="rigid_contact_forces",
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
    u_bounds["tau"] = [-500.0], [500.0]
    u_bounds["muscles"] = [0.0] * bio_model.nb_muscles, [1.0] * bio_model.nb_muscles

    u_init = InitialGuessList()
    u_init["tau"] = [1.0]
    u_init["muscles"] = [0.5] * bio_model.nb_muscles

    # Define algebraic states path constraint
    a_bounds = BoundsList()
    # Do not pull on the ground only push
    a_bounds["rigid_contact_forces"] = [-200.0, 0.0, 0.0], [200.0, 200.0, 200.0]

    a_init = InitialGuessList()
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
        variable_mappings=dof_mapping,
        objective_functions=objective_functions,
        constraints=constraints,
        multinode_constraints=multinode_constraints,
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3, defects_type=DefectType.IMPLICIT),
    )


def main():
    biorbd_model_path = "models/2segments_4dof_2contacts_1muscle.bioMod"
    t = 0.3
    ns = 10
    ocp = prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        phase_time=t,
        n_shooting=ns,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
    nlp = ocp.nlp[0]

    time = np.reshape(sol.decision_time(to_merge=SolutionMerge.NODES), (-1,))
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    algebraic_states = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)
    q, qdot, tau_residual, mus, contact_forces = (
        states["q"],
        states["qdot"],
        controls["tau"],
        controls["muscles"],
        algebraic_states["rigid_contact_forces"],
    )

    # --- Get contact position --- #
    contact_positions = np.zeros((2, 3, time.shape[0]))
    for i_node in range(time.shape[0]):
        for i_contact in range(2):
            contact_positions[i_contact, :, i_node] = np.reshape(
                nlp.model.rigid_contact_position(i_contact)(q[:, i_node], []), (3,)
            )

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

    names_contact_forces = ocp.nlp[0].model.rigid_contact_names
    for i_ax, elt in enumerate(contact_forces):
        axs[1].plot(time, elt, ".-", label=f"{names_contact_forces[i_ax]}")
    axs[1].legend()
    axs[1].grid()
    axs[1].set_title("Contact forces [N]")
    plt.savefig("test.png")
    plt.show()

    # --- Show results --- #
    sol.animate(viewer="pyorerun")


if __name__ == "__main__":
    main()
