"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement
and superimpose the same corner to a different marker at the end.
It is designed to show how one can define its own custom dynamics function if the provided ones are not
sufficient.

More specifically this example reproduces the behavior of the DynamicsFcn.TORQUE_DRIVEN using a custom dynamics
"""
import numpy as np

from biorbd import marker_index
from casadi import MX, SX, Function, jacobian, vertcat

from bioptim import (
    Bounds,
    BiMappingList,
    BiorbdModelHolonomic,
    ConfigureProblem,
    ConstraintList,
    DynamicsEvaluation,
    DynamicsFunctions,
    DynamicsList,
    InitialGuess,
    NonLinearProgram,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OdeSolverBase,
    OptimalControlProgram,
    Solver,
)


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

    u = DynamicsFunctions.get(nlp.states["u"], states)
    udot = DynamicsFunctions.get(nlp.states["udot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    uddot = nlp.model.partitioned_forward_dynamics(u, udot, tau)

    return DynamicsEvaluation(dxdt=vertcat(udot, uddot), defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """

    name_u = [nlp.model.name_dof[i] for i in range(nlp.model.nb_independent_joints)]
    axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, "u")
    ConfigureProblem.configure_new_variable("u", name_u, ocp, nlp, True, False, False, axes_idx=axes_idx)

    name = "udot"
    name_qdot = ConfigureProblem._get_kinematics_based_names(nlp, "qdot")
    name_udot = [name_qdot[i] for i in range(nlp.model.nb_independent_joints)]
    axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
    ConfigureProblem.configure_new_variable(name, name_udot, ocp, nlp, True, False, False, axes_idx=axes_idx)

    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic, expand=False)


def generate_close_loop_constraint(
    biorbd_model, marker_1: str, marker_2: str, index: slice = slice(0, 3), local_frame_index: int = None
) -> tuple[Function, Function, Function]:
    """Generate a close loop constraint between two markers"""

    # symbolic variables to create the functions
    q_sym = MX.sym("q", biorbd_model.nb_q, 1)
    q_dot_sym = MX.sym("q_dot", biorbd_model.nb_qdot, 1)
    q_ddot_sym = MX.sym("q_ddot", biorbd_model.nb_qdot, 1)

    # symbolic markers in global frame
    marker_1_sym = biorbd_model.marker(q_sym, index=marker_index(biorbd_model.model, marker_1))
    marker_2_sym = biorbd_model.marker(q_sym, index=marker_index(biorbd_model.model, marker_2))

    # if local frame is provided, the markers are expressed in the same local frame
    if local_frame_index is not None:
        jcs_t = biorbd_model.homogeneous_matrices_in_global(q_sym, local_frame_index, inverse=True)
        marker_1_sym = (jcs_t.to_mx() @ vertcat(marker_1_sym, 1))[:3]
        marker_2_sym = (jcs_t.to_mx() @ vertcat(marker_2_sym, 1))[:3]

    # the constraint is the distance between the two markers, set to zero
    constraint = (marker_1_sym - marker_2_sym)[index]
    # the jacobian of the constraint
    constraint_jacobian = jacobian(constraint, q_sym)

    constraint_func = Function(
        "holonomic_constraint",
        [q_sym],
        [constraint],
        ["q"],
        ["holonomic_constraint"],
    ).expand()

    constraint_jacobian_func = Function(
        "holonomic_constraint_jacobian",
        [q_sym],
        [constraint_jacobian],
        ["q"],
        ["holonomic_constraint_jacobian"],
    ).expand()

    # the double derivative of the constraint
    constraint_double_derivative = (
        constraint_jacobian_func(q_sym) @ q_ddot_sym + constraint_jacobian_func(q_dot_sym) @ q_dot_sym
    )

    constraint_double_derivative_func = Function(
        "holonomic_constraint_double_derivative",
        [q_sym, q_dot_sym, q_ddot_sym],
        [constraint_double_derivative],
        ["q", "q_dot", "q_ddot"],
        ["holonomic_constraint_double_derivative"],
    ).expand()

    return constraint_func, constraint_jacobian_func, constraint_double_derivative_func


def prepare_ocp(
    biorbd_model_path: str,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    custom_dynamics: bool = True,
    n_shooting: int = 100,
    final_time: float = 1,
    use_sx: bool = False,
) -> (OptimalControlProgram, BiorbdModelHolonomic):
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    ode_solver: OdeSolverBase
        The type of ode solver used
    custom_dynamics: bool
        If the user wants to use custom dynamics
    n_shooting: int
        The number of shooting points
    final_time: float
        The time at the final node

    Returns
    -------
    The ocp ready to be solved
    """

    # --- Options --- #
    # BioModel path
    bio_model = BiorbdModelHolonomic(biorbd_model_path)
    constraint, constraint_jacobian, constraint_double_derivative = generate_close_loop_constraint(
        bio_model, "marker_1", "marker_3", index=slice(1, 3), local_frame_index=0
    )
    #
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
    mapping.add("q", [0, None, None, 1], [0, 3])
    mapping.add("qdot", [0, None, None, 1], [0, 3])
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"], mapping=mapping)

    # Initial guess
    q_t0 = np.array([1.54, 1.54])  # change this line for automatic initial guess that satisfies the constraints
    # Translations between Seg0 and Seg1 at t0, calculated with cos and sin as Seg1 has no parent
    all_q_t0 = np.array([q_t0[0], q_t0[1], 0, 0])

    x_init = InitialGuess(all_q_t0)
    x_bounds[:, 0] = all_q_t0
    x_bounds[0, -1] = -1.54
    x_bounds[1, -1] = 0

    # Define control path constraint
    tau_min, tau_max, tau_init = -100, 100, 0

    variable_bimapping = BiMappingList()

    variable_bimapping.add("tau", to_second=[0, None, None, 1], to_first=[0, 3])
    u_bounds = Bounds([tau_min] * 2, [tau_max] * 2)
    u_init = InitialGuess([tau_init] * 2)

    # ------------- #

    return (
        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
            constraints,
            ode_solver=ode_solver,
            use_sx=use_sx,
            assume_phase_dynamics=True,
            variable_mappings=variable_bimapping,
            n_threads=8,
        ),
        bio_model,
    )


def main():
    """
    Runs the optimization and animates it
    """

    model_path = "models/two_pendulums.bioMod"
    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path, n_shooting=10, final_time=1)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=500))

    # --- Show results --- #
    # sol.animate()
    q = np.zeros((4, 11))
    for i, ui in enumerate(sol.states["u"].T):
        vi = bio_model.compute_v_from_u_numeric(ui, v_init=np.zeros(2)).toarray()
        qi = bio_model.q_from_u_and_v(ui[:, np.newaxis], vi).toarray().squeeze()
        q[:, i] = qi

    np.set_printoptions(precision=15)
    print(q[:, -1])
    import bioviz

    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()


if __name__ == "__main__":
    main()
