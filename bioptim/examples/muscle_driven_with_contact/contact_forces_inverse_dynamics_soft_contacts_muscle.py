"""
This example shows how to impose the dynamics through an inverse dynamics defect in collocation.
It also shows how to impose the contact forces as an implicit constraints.
"""

from matplotlib import pyplot as plt
import numpy as np
from casadi import MX, SX, vertcat
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
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
)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None):
    # Usual variables
    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False, as_states_dot=True)
    ConfigureProblem.configure_qddot(ocp, nlp, as_states=False, as_controls=False, as_states_dot=True)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)  # Residual torques
    ConfigureProblem.configure_muscles(ocp, nlp, as_states=False, as_controls=True)  # Muscle activation

    # Implicit variables
    ConfigureProblem.configure_soft_contact_forces(
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
    soft_contact_forces = nlp.algebraic_states["soft_contact_forces"].cx

    # Map to external forces
    external_forces = MX.zeros(9 * 2)
    external_forces[0:6] = soft_contact_forces[0:6]
    external_forces[9:15] = soft_contact_forces[6:12]

    # Compute joint torques
    muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations)
    tau = muscles_tau + residual_tau

    # Defects
    slope_q = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
    slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.scaled.cx)

    # qdot
    qdot_defect = qdot - slope_q
    # qddot
    tau_id = DynamicsFunctions.inverse_dynamics(
        nlp, q, slope_q, slope_qdot, with_contact=False, external_forces=external_forces
    )
    tau_defect = tau - tau_id
    # contact
    contact_forces = nlp.model.soft_contact_forces()(q, qdot, nlp.parameters.cx)
    contact_defect = soft_contact_forces - contact_forces
    return DynamicsEvaluation(dxdt=None, defects=vertcat(qdot_defect, tau_defect, contact_defect))


def prepare_ocp(biorbd_model_path, phase_time, n_shooting, expand_dynamics=True):

    # Indicate to the model creator that there will be two rigid contacts in the form of optimization variables
    external_force_set = ExternalForceSetVariables()
    external_force_set.add(force_name="Seg1_contact1", segment="Seg1", use_point_of_application=False)
    external_force_set.add(force_name="Seg1_contact2", segment="Seg1", use_point_of_application=False)

    # BioModel
    bio_model = BiorbdModel(biorbd_model_path, external_force_set=external_force_set)

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
    )

    multinode_constraints = MultinodeConstraintList()
    for i_node in range(n_shooting):
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
        dynamics,
        n_shooting,
        phase_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        a_bounds=a_bounds,
        x_init=x_init,
        u_init=u_init,
        a_init=a_init,
        control_type=ControlType.LINEAR_CONTINUOUS,
        objective_functions=objective_functions,
        multinode_constraints=multinode_constraints,
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3, defects_type=DefectType.IMPLICIT),
    )


def main():
    biorbd_model_path = "models/2segments_4dof_2soft_contacts_1muscle.bioMod"
    t = 1
    ns = 100
    ocp = prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        phase_time=t,
        n_shooting=ns,
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(1000)
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
    plt.savefig("test.png")
    plt.show()

    # --- Reintegrate solution --- #
    # TODO: Charbie -> This is not working yet
    sol.integrate(
        shooting_type=Shooting.SINGLE,
        integrator=SolutionIntegrator.SCIPY_DOP853,
        to_merge=SolutionMerge.NODES,
        return_time=True,
    )

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
