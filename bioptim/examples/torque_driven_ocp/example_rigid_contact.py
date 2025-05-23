"""
A very simple optimal control program where a 2D leg want to jump as high as possible by pushing on the ground.
"""

from operator import index

import matplotlib.pyplot as plt
import numpy as np
from casadi import MX
from scipy.odr import quadratic

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Node,
    Solver,
    Shooting,
    Solution,
    SolutionIntegrator,
    PhaseDynamics,
    SolutionMerge,
    DefectType,
    Axis,
    ContactType,
    PenaltyController,
    ExternalForceSetVariables,
)


def custom_com_over_contact(controller: PenaltyController) -> MX:
    marker_position = controller.model.marker(0)(controller.states["q"].cx, controller.parameters.cx)
    com_position = controller.model.center_of_mass()(controller.states["q"].cx, controller.parameters.cx)
    return com_position[1] - marker_position[1]


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    ode_solver: OdeSolverBase,
    contact_type: list[ContactType],
    n_threads: int = 8,
    use_sx: bool = False,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    if (
        ContactType.RIGID_IMPLICIT in contact_type
        or ContactType.SOFT_IMPLICIT in contact_type
        or ContactType.SOFT_EXPLICIT in contact_type
    ):
        # Indicate to the model creator that there will be two rigid contacts in the form of optimization variables
        external_force_set = ExternalForceSetVariables()
        external_force_set.add(force_name="Seg2_contact0", segment="Seg2", use_point_of_application=True)

        # BioModel
        bio_model = BiorbdModel(biorbd_model_path, external_force_set=external_force_set)
    else:
        # BioModel
        bio_model = BiorbdModel(biorbd_model_path)

    tau_min, tau_max = -100, 100

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-1, axes=Axis.Z, quadratic=False
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, axes=Axis.Z, quadratic=False
    )
    objective_functions.add(
        custom_com_over_contact, custom_type=ObjectiveFcn.Mayer, node=Node.START, quadratic=True, weight=1000
    )

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN,
        contact_type=contact_type,
        phase_dynamics=phase_dynamics,
        ode_solver=ode_solver,
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, marker_index=0, node=Node.START)

    if ContactType.RIGID_EXPLICIT in contact_type or ContactType.SOFT_EXPLICIT in contact_type:
        constraints.add(
            ConstraintFcn.TRACK_CONTACT_FORCES, node=Node.ALL_SHOOTING, contact_index=1, min_bound=0, max_bound=np.inf
        )
    elif ContactType.RIGID_IMPLICIT in contact_type:
        constraints.add(
            ConstraintFcn.TRACK_ALGEBRAIC_STATE,
            node=Node.ALL_SHOOTING,
            key="rigid_contact_forces",
            index=1,
            min_bound=0,
            max_bound=np.inf,
        )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS, marker_index=1, node=Node.ALL_SHOOTING, min_bound=0, max_bound=np.inf, axes=Axis.Z
    )
    constraints.add(
        ConstraintFcn.TRACK_MARKERS, marker_index=2, node=Node.ALL_SHOOTING, min_bound=0, max_bound=np.inf, axes=Axis.Z
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")

    x_bounds["q"].min[:, 0] = [-0.1, 0.1, -3 * np.pi / 4, np.pi / 4, -3 * np.pi / 4]
    x_bounds["q"].max[:, 0] = [0.1, 0.1, -np.pi / 4, 3 * np.pi / 4, -np.pi / 4]
    x_bounds["qdot"][:, 0] = [0, 0, 0, 0, 0]

    # Initial guess
    x_init = InitialGuessList()
    x_init["q"] = [0, 0, -np.pi / 4, np.pi / 2, -np.pi / 2]
    x_init["qdot"] = [0, 0, 0, 0, 0]

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
        constraints=constraints,
        use_sx=use_sx,
        n_threads=n_threads,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """
    biorbd_model_path = "../torque_driven_ocp/models/3segments_4dof_1contact.bioMod"
    n_shooting = 30
    final_time = 1
    defect_type = [DefectType.QDOT_EQUALS_POLYNOMIAL_SLOPE, DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS]
    ode_solver = OdeSolver.COLLOCATION(polynomial_degree=5, defects_type=defect_type)
    contact_type = [ContactType.RIGID_IMPLICIT]

    # Prepare OCP to reach the second marker
    ocp = prepare_ocp(biorbd_model_path, n_shooting, final_time, ode_solver, contact_type)

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    sol = ocp.solve(solver)

    nlp = ocp.nlp[0]
    time = sol.decision_time(to_merge=SolutionMerge.NODES)
    q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
    qdot = sol.decision_states(to_merge=SolutionMerge.NODES)["qdot"]

    # --- Plot the reintegration -- #
    sol_integrated = sol.integrate(
        shooting_type=Shooting.SINGLE,
        integrator=SolutionIntegrator.SCIPY_DOP853,
        to_merge=SolutionMerge.NODES,
        return_time=False,
    )
    time_integrated = np.linspace(0, time[-1], sol_integrated["q"].shape[1])
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
    plt.savefig(f"reintegration_{defect_type.value}_{contact_type}.png")
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
