"""
This example is a clone of the getting_started/pendulum.py example with the difference that the
model now evolves in an environment where the gravity can be modified.
The goal of the solver it to find the optimal gravity (target = 8 N/kg), while performing the
pendulum balancing task
It is designed to show how one can define its own parameter objective functions if the provided ones are not
sufficient.
"""

from typing import Any

import numpy as np
from casadi import MX
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    InterpolationType,
    ParameterList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    ParameterObjectiveList,
    PenaltyController,
    ObjectiveList,
    PhaseDynamics,
    VariableScaling,
    VariableScalingList,
)


def my_parameter_function(bio_model: BiorbdModel, value: MX, extra_value: Any):
    """
    The pre dynamics function is called right before defining the dynamics of the system. If one wants to
    modify the dynamics (e.g. optimize the gravity in this case), then this function is the proper way to do it.

    Parameters
    ----------
    bio_model: BiorbdModel
        The model to modify by the parameters
    value: MX
        The CasADi variables to modify the model
    extra_value: Any
        Any parameters required by the user. The name(s) of the extra_value must match those used in parameter.add
    """

    value[2] *= extra_value
    bio_model.set_gravity(value)


def set_mass(bio_model: BiorbdModel, value: MX):
    """
    The pre dynamics function is called right before defining the dynamics of the system. If one wants to
    modify the dynamics (e.g. optimize the gravity in this case), then this function is the proper way to do it.

    Parameters
    ----------
    bio_model: BiorbdModel
        The model to modify by the parameters
    value: MX
        The CasADi variables to modify the model
    """

    bio_model.segments[0].characteristics().setMass(value)


def my_target_function(controller: PenaltyController, key: str) -> MX:
    """
    The target function is a penalty function.

    Parameters
    ----------
    controller: PenaltyController
        A reference the controller
    key: str
        A variable defined by the user when declaring the custom function
    Returns
    -------
    The value to minimize. If a target value exist (target parameters) it is automatically added, and therefore
    should not be added by hand here (that is, the next line should not read: return value - target)
    """
    return controller.parameters[key].cx


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    optim_gravity: bool,
    optim_mass: bool,
    min_g: np.ndarray,
    max_g: np.ndarray,
    target_g: np.ndarray,
    min_m: float,
    max_m: float,
    target_m: float,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = False,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    optim_gravity: bool
        If the gravity should be optimized
    optim_mass: bool
        If the mass should be optimized
    min_g: np.ndarray
        The minimal value for the gravity
    max_g: np.ndarray
        The maximal value for the gravity
    target_g: np.ndarray
        The target value for the gravity
    min_m: float
        The minimal value for the mass
    max_m: float
        The maximal value for the mass
    target_m: float
        The target value for the mass
    ode_solver: OdeSolverBase
        The type of ode solver used
    use_sx: bool
        If the program should be constructed using SX instead of MX (longer to create the CasADi graph, faster to solve)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The ocp ready to be solved
    """

    # Define the parameter to optimize
    parameters = ParameterList(use_sx=use_sx)
    parameter_objectives = ParameterObjectiveList()
    parameter_bounds = BoundsList()
    parameter_init = InitialGuessList()

    if optim_gravity:
        g_scaling = VariableScaling("gravity_xyz", np.array([1, 1, 10.0]))
        parameters.add(
            "gravity_xyz",  # The name of the parameter
            my_parameter_function,  # The function that modifies the biorbd model
            size=3,  # The number of elements this particular parameter vector has
            scaling=g_scaling,  # The scaling of the parameter
            extra_value=1,  # You can define as many extra arguments as you want
        )
        # TODO: Add the capability to send values to the parameter so instead of being a MX it is a float
        # This would look more or less like the following (with the values the dispatch function similar to the one in Penalty).
        # This would simultanously solve the dispatching of all forces and the phase parameters
        # parameters.add(
        #     "gravity_xyz",  # The name of the parameter
        #     my_parameter_function,  # The function that modifies the biorbd model
        #     values=lambda phase_idx, node_idx=...
        # )

        # Give the parameter some min and max bounds
        parameter_bounds.add("gravity_xyz", min_bound=min_g, max_bound=max_g, interpolation=InterpolationType.CONSTANT)

        # and an initial condition
        parameter_init["gravity_xyz"] = (min_g + max_g) / 2

        # and an objective function
        parameter_objectives.add(
            my_target_function,
            weight=1000,
            quadratic=True,
            custom_type=ObjectiveFcn.Parameter,
            target=target_g / g_scaling.scaling,  # Make sure your target fits the scaling
            key="gravity_xyz",
        )

    if optim_mass:
        m_scaling = VariableScaling("mass", np.array([10.0]))
        parameters.add(
            "mass",  # The name of the parameter
            set_mass,  # The function that modifies the biorbd model
            size=1,  # The number of elements this particular parameter vector has
            scaling=m_scaling,  # The scaling of the parameter
        )

        parameter_bounds.add("mass", min_bound=[min_m], max_bound=[max_m], interpolation=InterpolationType.CONSTANT)

        parameter_init["mass"] = (min_m + max_m) / 2

        parameter_objectives.add(
            my_target_function,
            weight=10000,
            quadratic=True,
            custom_type=ObjectiveFcn.Parameter,
            target=target_m / m_scaling.scaling.T,  # Make sure your target fits the scaling
            key="mass",
        )

    # --- Options --- #
    bio_model = BiorbdModel(biorbd_model_path, parameters=parameters)
    n_tau = bio_model.nb_tau

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1)

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics, ode_solver=ode_solver
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 3.14
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    # Define control path constraint
    tau_min, tau_max = -300, 300
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau
    u_bounds["tau"][1, :] = 0

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        parameters=parameters,
        parameter_objectives=parameter_objectives,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        use_sx=use_sx,
    )


def main():
    """
    Solve and print the optimized value for the gravity and animate the solution
    """
    optim_gravity = True
    optim_mass = True
    ocp = prepare_ocp(
        biorbd_model_path="models/pendulum.bioMod",
        final_time=3,
        n_shooting=100,
        optim_gravity=optim_gravity,
        optim_mass=optim_mass,
        min_g=np.array([-1, -1, -10]),
        max_g=np.array([1, 1, -5]),
        min_m=10,
        max_m=30,
        target_g=np.array([0, 0, -9.81])[:, np.newaxis],
        target_m=20,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Get the results --- #
    if optim_gravity:
        gravity = sol.parameters["gravity_xyz"]
        print(f"Optimized gravity: {gravity}")

    if optim_mass:
        mass = sol.parameters["mass"]
        print(f"Optimized mass: {mass}")

    # --- Show results --- #
    # sol.graphs()
    sol.animate(n_frames=200)


if __name__ == "__main__":
    main()
