"""
This example shows how to impose the dynamics through an inverse dynamics defect in collocation.
It also shows how to impose the contact forces as an implicit constraint.
Introducing the rigid contact force derivatives allows to reach convergence of the problem.
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
    ContactType,
)


def custom_configure(
    ocp: OptimalControlProgram,
    nlp: NonLinearProgram,
    numerical_data_timeseries=None,
    contact_types: list[ContactType] = [],
):
    # Usual variables
    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)  # Residual torques
    ConfigureProblem.configure_muscles(ocp, nlp, as_states=False, as_controls=True)  # Muscle activation
    ConfigureProblem.configure_rigid_contact_forces(
        ocp, nlp, as_states=True, as_algebraic_states=False, as_controls=True
    )
    name_contact_forces = [name for name in nlp.model.rigid_contact_names]
    ConfigureProblem.configure_new_variable(
        "rigid_contact_forces_derivatives", name_contact_forces, ocp, nlp, as_states=False, as_controls=True
    )

    # Dynamics
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics, contact_types=contact_types)


def custom_dynamics(
    time: MX | SX,
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    algebraic_states: MX | SX,
    numerical_timeseries: MX | SX,
    nlp: NonLinearProgram,
    contact_types: list[ContactType],
) -> DynamicsEvaluation:
    """
    The defects are only evaluated at the collocation nodes, but not at the first node.
    """
    # Variables
    q = DynamicsFunctions.get(nlp.states["q"], nlp.states.scaled.cx)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], nlp.states.scaled.cx)
    # rigid_contact_forces = DynamicsFunctions.get(
    #     nlp.states["rigid_contact_forces"], nlp.states.scaled.cx
    # )  # Extracterd automatically in inverse_dynamics
    residual_tau = DynamicsFunctions.get(nlp.controls["tau"], nlp.controls.scaled.cx)
    mus_activations = DynamicsFunctions.get(nlp.controls["muscles"], nlp.controls.scaled.cx)
    rigid_contact_forces_derivatives = DynamicsFunctions.get(
        nlp.controls["rigid_contact_forces_derivatives"], nlp.controls.scaled.cx
    )

    # Compute joint torques
    muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations)
    tau = muscles_tau + residual_tau

    dq = qdot
    ddq = nlp.model.forward_dynamics(with_contact=False)(q, qdot, tau, [], nlp.parameters.cx)
    drigid_contact_forces = rigid_contact_forces_derivatives
    dxdt = vertcat(dq, ddq, drigid_contact_forces)

    defects = None
    if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
        # Defects
        slope_q = DynamicsFunctions.get(nlp.states_dot["q"], nlp.states_dot.scaled.cx)
        slope_qdot = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
        slope_contacts = DynamicsFunctions.get(nlp.states_dot["rigid_contact_forces"], nlp.states_dot.scaled.cx)
        tau_id = DynamicsFunctions.inverse_dynamics(nlp, q, qdot, slope_qdot, contact_types=contact_types)

        # qdot
        defects = vertcat(qdot - slope_q, tau - tau_id, rigid_contact_forces_derivatives - slope_contacts)

    return DynamicsEvaluation(dxdt=dxdt, defects=defects)


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


def prepare_ocp(biorbd_model_path, phase_time, n_shooting, expand_dynamics=True):

    # BioModel
    bio_model = BiorbdModel(biorbd_model_path, contact_types=[ContactType.RIGID_IMPLICIT])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, index=[1, 2, 3])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, weight=100)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        custom_configure,
        dynamic_function=custom_dynamics,
        expand_dynamics=expand_dynamics,
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3),
    )

    # Constraints
    constraints = ConstraintList()
    # This constraint is necessary to prevent the contacts from drifting
    constraints.add(
        contact_velocity_all_points,
        node=Node.ALL_SHOOTING,
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
    x_bounds.add(
        "rigid_contact_forces",
        min_bound=[-200.0, 0.0, 0.0],
        max_bound=[200.0, 200.0, 200.0],
        interpolation=InterpolationType.CONSTANT,
    )

    # Initial guess
    x_init = InitialGuessList()
    x_init["q"] = pose_at_first_node
    x_init["qdot"] = np.zeros((n_qdot,))
    x_init["rigid_contact_forces"] = [0.0, 0.0, 0.0]

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [-200.0] * bio_model.nb_tau, [200.0] * bio_model.nb_tau
    u_bounds["muscles"] = [0.0] * bio_model.nb_muscles, [1.0] * bio_model.nb_muscles
    u_bounds.add(
        "rigid_contact_forces_derivatives",
        min_bound=[-200.0, -200.0, -200.0],
        max_bound=[200.0, 200.0, 200.0],
        interpolation=InterpolationType.CONSTANT,
    )

    u_init = InitialGuessList()
    u_init["tau"] = [1.0] * bio_model.nb_tau
    u_init["muscles"] = [0.5] * bio_model.nb_muscles
    u_init["rigid_contact_forces"] = [1.0, 1.0, 1.0]

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        phase_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        control_type=ControlType.LINEAR_CONTINUOUS,
        objective_functions=objective_functions,
        constraints=constraints,
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
    # ocp.add_plot_penalty()

    # --- Solve the program --- #
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(10000)
    solver.set_tol(1e-4)
    sol = ocp.solve(solver)
    nlp = ocp.nlp[0]
    sol.graphs()

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
    for i_ax in range(len(names_contact_forces)):
        axs[1].plot(time, contact_forces[i_ax, :], ".-", label=f"{names_contact_forces[i_ax]}")
    axs[1].legend()
    axs[1].grid()
    axs[1].set_title("Contact forces [N]")
    plt.savefig("test.png")
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
