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
)

def custom_configure(
    ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None
):
    # Usual variables
    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False, as_states_dot=True)
    ConfigureProblem.configure_qddot(ocp, nlp, as_states=False, as_controls=False, as_states_dot=True)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)  # Residual torques
    ConfigureProblem.configure_muscles(ocp, nlp, as_states=False, as_controls=True)  # Muscle activation

    # Implicit variables
    ConfigureProblem.configure_rigid_contact_forces(ocp, nlp, as_states=False, as_algebraic_states=True,
                                                    as_controls=False)

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
    my_additional_factor=1,
) -> DynamicsEvaluation:

    # Variables
    q = nlp.get_var_from_states_or_controls("q", states, controls)
    qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
    residual_tau = nlp.get_var_from_states_or_controls("tau", states, controls)
    mus_activations = nlp.get_var_from_states_or_controls("muscles", states, controls)

    # Get external forces from algebraic states
    rigid_contact_forces = nlp.get_external_forces("rigid_contact_forces", states, controls, algebraic_states,
                                              numerical_timeseries)
    # Map to external forces
    external_forces = nlp.model.map_rigid_contact_forces_to_global_forces(rigid_contact_forces, q, parameters)


    # Compute joint torques
    muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations)
    tau = muscles_tau + residual_tau

    # Defects
    slope_q = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
    slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.scaled.cx)
    tau_id = DynamicsFunctions.inverse_dynamics(nlp, q, slope_q, slope_qdot, with_contact=False,
                                                external_forces=external_forces)
    defects = vertcat(qdot - slope_q, tau - tau_id)

    return DynamicsEvaluation(dxdt=None, defects=defects)

def contact_velocity(controller):
    contact_velocities = []
    for i_contact in range(2):
        qs = horzcat(*([controller.states["q"].cx_start] + controller.states["q"].cx_intermediates))
        qdots = horzcat(*([controller.states["qdot"].cx_start] + controller.states["qdot"].cx_intermediates))
        for i_sn in range(len(qs)):
            contact_velocity = controller.model.contact_velocity(i_contact)(qs[i_sn], qdots[i_sn])
            contact_velocities += [contact_velocity]
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

    tau_min, tau_max, tau_init = -500.0, 500.0, 0.0
    activation_min, activation_max, activation_init = 0.0, 1.0, 0.5

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT)

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

    # Path constraint
    n_q = bio_model.nb_q
    n_qdot = n_q
    n_contacts = 3
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
    u_bounds["tau"] = [tau_min] * len(dof_mapping["tau"].to_first), [tau_max] * len(dof_mapping["tau"].to_first)
    u_bounds["muscles"] = [activation_min] * bio_model.nb_muscles, [activation_max] * bio_model.nb_muscles

    u_init = InitialGuessList()
    u_init["tau"] = [tau_init] * len(dof_mapping["tau"].to_first)
    u_init["muscles"] = [activation_init] * bio_model.nb_muscles

    # Define algebraic states path constraint
    a_bounds = BoundsList()
    # Do not pull on the ground only push
    a_bounds["rigid_contact_forces"] = [-200.0, 0.0, 0.0], [200.0, 200.0, 200.0]

    a_init = InitialGuessList()
    a_init["tau"] = [0.0, 0.0, 0.0]

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
        objective_functions=objective_functions,
        constraints=constraints,
        variable_mappings=dof_mapping,
    )


def main():
    biorbd_model_path = "models/2segments_4dof_2contacts_1muscle.bioMod"
    t = 0.3
    ns = 10
    dt = t/ns
    ocp = prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        phase_time=t,
        n_shooting=ns,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    nlp = ocp.nlp[0]
    nlp.model = BiorbdModel(biorbd_model_path)

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau, mus = states["q"], states["qdot"], controls["tau"], controls["muscles"]

    x = np.concatenate((q, qdot))
    u = np.concatenate((tau, mus))
    contact_forces = np.zeros((3, nlp.ns))
    for i_node in range(nlp.ns):
        contact_forces[:, i_node] = np.reshape(np.array(nlp.contact_forces_func([dt*i_node, dt*(i_node+1)], x[:, i_node], u[:, i_node], [], [], [])), (3, ))

    names_contact_forces = ocp.nlp[0].model.rigid_contact_names
    for i, elt in enumerate(contact_forces):
        plt.plot(np.linspace(0, t, ns + 1)[:-1], elt, ".-", label=f"{names_contact_forces[i]}")
    plt.legend()
    plt.grid()
    plt.title("Contact forces")
    plt.show()

    # --- Show results --- #
    sol.animate(viewer="pyorerun")


if __name__ == "__main__":
    main()
