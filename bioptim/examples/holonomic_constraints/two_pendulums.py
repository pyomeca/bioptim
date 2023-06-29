"""
This example presents how to implement a holonomic constraint in bioptim.
The simulation is two single pendulum that are forced to be coherent with a holonomic constraint. It is then a double
pendulum simulation.
"""
import numpy as np

import bioviz

from casadi import MX, SX, vertcat

from bioptim import (
    BiMappingList,
    HolonomicBiorbdModel,
    BoundsList,
    ConfigureProblem,
    ConstraintList,
    DynamicsEvaluation,
    DynamicsFunctions,
    DynamicsList,
    InitialGuessList,
    NonLinearProgram,
    ObjectiveFcn,
    ObjectiveList,
    OptimalControlProgram,
    Solver,
)

from bioptim.examples.discrete_mechanics_and_optimal_control.holonomic_constraints import HolonomicConstraintFcn


def custom_dynamic(
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(x, u, p)

    Parameters
    ----------
    states: MX | SX
        The state of the system
    controls: MX | SX
        The controls of the system
    parameters: MX | SX
        The parameters acting on the system
    nlp: NonLinearProgram
        A reference to the phase

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    q_u = DynamicsFunctions.get(nlp.states["q_u"], states)
    qdot_u = DynamicsFunctions.get(nlp.states["qdot_u"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    qddot_u = nlp.model.partitioned_forward_dynamics(q_u, qdot_u, tau)

    return DynamicsEvaluation(dxdt=vertcat(qdot_u, qddot_u), defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    """
    Tell the program which variables are states and controls.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """

    name = "q_u"
    names_u = [nlp.model.name_dof[nlp.model.independent_joint_index[i]] for i in range(nlp.model.nb_independent_joints)]
    axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
    ConfigureProblem.configure_new_variable(name, names_u, ocp, nlp, True, False, False, axes_idx=axes_idx)

    name = "qdot_u"
    names_qdot = ConfigureProblem._get_kinematics_based_names(nlp, "qdot")
    names_udot = [names_qdot[nlp.model.independent_joint_index[i]] for i in range(nlp.model.nb_independent_joints)]
    axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
    ConfigureProblem.configure_new_variable(name, names_udot, ocp, nlp, True, False, False, axes_idx=axes_idx)

    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic, expand=False)


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int = 100,
    final_time: float = 1,
) -> (HolonomicBiorbdModel, OptimalControlProgram):
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    n_shooting: int
        The number of shooting points
    final_time: float
        The time at the final node

    Returns
    -------
    The ocp ready to be solved
    """
    bio_model = HolonomicBiorbdModel(biorbd_model_path)
    constraint, constraint_jacobian, constraint_double_derivative = HolonomicConstraintFcn.superimpose_markers(
        bio_model, "marker_1", "marker_3", index=slice(1, 3), local_frame_index=0
    )
    bio_model.add_holonomic_constraint(
        constraint=constraint,
        constraint_jacobian=constraint_jacobian,
        constraint_double_derivative=constraint_double_derivative,
    )
    bio_model.set_dependencies(independent_joint_index=[0, 3], dependent_joint_index=[1, 2])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=0.5, max_bound=0.6)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=custom_dynamic, expand=False)

    # Path Constraints
    constraints = ConstraintList()

    # Boundaries
    mapping = BiMappingList()
    mapping.add("q", to_second=[0, None, None, 1], to_first=[0, 3])
    mapping.add("qdot", to_second=[0, None, None, 1], to_first=[0, 3])
    x_bounds = BoundsList()
    x_bounds["q_u"] = bio_model.bounds_from_ranges("q", mapping=mapping)
    x_bounds["qdot_u"] = bio_model.bounds_from_ranges("qdot", mapping=mapping)

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q_u", [1.54, 1.54])
    x_init.add("qdot_u", [0, 0])
    x_bounds["q_u"][:, 0] = [1.54, 1.54]
    x_bounds["qdot_u"][:, 0] = [0, 0]
    x_bounds["q_u"][0, -1] = -1.54
    x_bounds["q_u"][1, -1] = 0

    # Define control path constraint
    variable_bimapping = BiMappingList()
    tau_min, tau_max, tau_init = -100, 100, 0
    variable_bimapping.add("tau", to_second=[0, None, None, 1], to_first=[0, 3])
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * 2, max_bound=[tau_max] * 2)
    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * 2)

    # ------------- #

    return (
        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_init=x_init,
            u_init=u_init,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            assume_phase_dynamics=True,
            variable_mappings=variable_bimapping,
            constraints=constraints,
        ),
        bio_model,
    )


def main():
    """
    Runs the optimization and animates it
    """

    model_path = "models/two_pendulums.bioMod"
    n_shooting = 100
    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path, n_shooting=n_shooting)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=500))  # show_online_optim not working yet

    # --- Show results --- #
    q = np.zeros((4, n_shooting + 1))
    for i, ui in enumerate(sol.states["q_u"].T):
        vi = bio_model.compute_v_from_u_numeric(ui, v_init=np.zeros(2)).toarray()
        qi = bio_model.q_from_u_and_v(ui[:, np.newaxis], vi).toarray().squeeze()
        q[:, i] = qi

    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()


if __name__ == "__main__":
    main()
