"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement
and superimpose the same corner to a different marker at the end.
It is designed to investigate the different way to define the initial guesses at each node sent to the solver

All the types of interpolation are shown:
InterpolationType.CONSTANT: All the values are the same at each node
InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT: Same as constant, but have the first
    and last nodes different. This is particularly useful when you want to fix the initial and
    final position and leave the rest of the movement free.
InterpolationType.LINEAR: The values are linearly interpolated between the first and last nodes.
InterpolationType.EACH_FRAME: Each node values are specified
InterpolationType.SPLINE: The values are interpolated from the first to last node using a cubic spline
InterpolationType.CUSTOM: Provide a user-defined interpolation function
"""

import numpy as np
from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    Bounds,
    InitialGuess,
    InterpolationType,
    OdeSolver,
    OdeSolverBase,
    VariableScalingList,
    MagnitudeType,
)


def custom_init_func(
    current_shooting_point: int, my_values: np.ndarray, n_shooting_custom: int, **extra_params
) -> np.ndarray:
    """
    The custom function for the x and u initial guesses (this particular one mimics linear interpolation)

    Parameters
    ----------
    current_shooting_point: int
        The current point to return the value, it is defined between [0; n_shooting_custom] for the states
        and [0; n_shooting_custom[ for the controls
    my_values: np.ndarray
        The values provided by the user
    n_shooting_custom: int
        The number of shooting point

    Returns
    -------
    The vector value of the initial guess at current_shooting_point
    """

    # Linear interpolation created with custom function
    return my_values[:, 0] + (my_values[:, -1] - my_values[:, 0]) * current_shooting_point / n_shooting_custom


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    random_init: bool = False,
    initial_guess: InterpolationType = InterpolationType.CONSTANT,
    ode_solver: OdeSolverBase = OdeSolver.COLLOCATION(),
    assume_phase_dynamics: bool = True,
) -> OptimalControlProgram:
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
    random_init: bool
        If True, the initial guess will be randomized
    initial_guess: InterpolationType
        The type of interpolation to use for the initial guess
    ode_solver: OdeSolverBase
        The type of ode solver used
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The ocp ready to be solved
    """

    # --- Options --- #
    # BioModel path
    bio_model = BiorbdModel(biorbd_model_path)
    nq = bio_model.nb_q
    nqdot = bio_model.nb_qdot
    ntau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)

    # Dynamics
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")

    # Path constraint and control path constraints
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57
    u_bounds = Bounds([tau_min] * ntau, [tau_max] * ntau)

    # Initial guesses
    t = None
    extra_params_x = {}
    extra_params_u = {}
    if initial_guess == InterpolationType.CONSTANT:
        x = [0] * (nq + nqdot)
        u = [tau_init] * ntau
    elif initial_guess == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        x = np.array([[1.0, 0.0, 0.0, 0, 0, 0], [1.5, 0.0, 0.785, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T
        u = np.array([[1.45, 9.81, 2.28], [0, 9.81, 0], [-1.45, 9.81, -2.28]]).T
    elif initial_guess == InterpolationType.LINEAR:
        x = np.array([[1.0, 0.0, 0.0, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T
        u = np.array([[1.45, 9.81, 2.28], [-1.45, 9.81, -2.28]]).T
    elif initial_guess == InterpolationType.EACH_FRAME:
        x = np.random.random((nq + nqdot, n_shooting + 1))
        u = np.random.random((ntau, n_shooting))
    elif initial_guess == InterpolationType.ALL_POINTS:
        if ode_solver.is_direct_collocation:
            ode_solver: OdeSolver.COLLOCATION
            x = np.random.random((nq + nqdot, n_shooting * (ode_solver.polynomial_degree + 1) + 1))
            u = np.random.random((ntau, n_shooting))
        else:
            x = np.random.random((nq + nqdot, n_shooting + 1))
            u = np.random.random((ntau, n_shooting))
    elif initial_guess == InterpolationType.SPLINE:
        # Bound spline assume the first and last point are 0 and final respectively
        t = np.hstack((0, np.sort(np.random.random((3,)) * final_time), final_time))
        x = np.random.random((nq + nqdot, 5))
        u = np.random.random((ntau, 5))
    elif initial_guess == InterpolationType.CUSTOM:
        # The custom function refers to the one at the beginning of the file. It emulates a Linear interpolation
        x = custom_init_func
        u = custom_init_func
        extra_params_x = {"my_values": np.random.random((nq + nqdot, 2)), "n_shooting_custom": n_shooting}
        extra_params_u = {"my_values": np.random.random((ntau, 2)), "n_shooting_custom": n_shooting}
    else:
        raise RuntimeError("Initial guess not implemented yet")

    x_init = InitialGuess(x, t=t, interpolation=initial_guess, **extra_params_x)
    u_init = InitialGuess(u, t=t, interpolation=initial_guess, **extra_params_u)
    if random_init:
        x_init = x_init.add_noise(
            bounds=x_bounds,
            magnitude=1,
            magnitude_type=MagnitudeType.RELATIVE,
            n_shooting=n_shooting + 1,
            bound_push=0.1,
        )
        u_init = u_init.add_noise(
            bounds=u_bounds,
            n_shooting=n_shooting,
            bound_push=0.1,
        )

    # Variable scaling
    x_scaling = VariableScalingList()
    x_scaling.add("q", scaling=[1] * bio_model.nb_q)
    x_scaling.add("qdot", scaling=[1] * bio_model.nb_qdot)

    xdot_scaling = VariableScalingList()
    xdot_scaling.add("qdot", scaling=[1] * bio_model.nb_qdot)
    xdot_scaling.add("qddot", scaling=[1] * bio_model.nb_qddot)

    u_scaling = VariableScalingList()
    u_scaling.add("tau", scaling=[1] * bio_model.nb_tau)

    return OptimalControlProgram(
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
        x_scaling=x_scaling,
        xdot_scaling=xdot_scaling,
        u_scaling=u_scaling,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    Solve the program for all the InterpolationType available
    """

    ocp = None
    for initial_guess in InterpolationType:
        print(f"Solving problem using {initial_guess} initial guess")

        ocp = None
        try:
            ocp = prepare_ocp(
                "models/cube.bioMod", n_shooting=30, final_time=2, random_init=True, initial_guess=initial_guess
            )
        except ValueError as message:
            if str(message) == "InterpolationType.ALL_POINTS must only be used with direct collocation":
                # This is normal as ALL_POINTS cannot be used without collocations
                pass
            else:
                raise ValueError(message)

    sol = ocp.solve()
    print("\n")

    # Print the last solution
    sol.animate()


if __name__ == "__main__":
    main()
