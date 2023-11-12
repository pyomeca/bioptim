"""
This example is inspired from the clear pike circle gymnastics skill. It is composed of two pendulums
representing the trunk and legs segments (only the hip flexion is actuated). The objective is to minimize the
maximum torque of the hip flexion while performing the clear pike circle motion. The maximum torque is included to the
problem as a parameter, all torque interval re constrained to be smaller than this parameter, this parameter is the
minimized.
"""

import numpy as np
import biorbd_casadi as biorbd
from casadi import MX
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ConstraintList,
    BoundsList,
    InitialGuessList,
    Node,
    ObjectiveFcn,
    BiMappingList,
    ParameterList,
    InterpolationType,
    BiorbdModel,
    PenaltyController,
    ParameterObjectiveList,
    ParameterConstraintList,
    ConstraintFcn,
)


def custom_constraint_parameters(controller: PenaltyController) -> MX:
    tau = controller.controls["tau_joints"].cx_start
    max_param = controller.parameters["max_tau"].cx
    val = max_param - tau
    return val


def my_parameter_function(bio_model: biorbd.Model, value: MX):
    return


def prepare_ocp(
    bio_model_path: str = "models/double_pendulum.bioMod",
) -> OptimalControlProgram:
    bio_model = (BiorbdModel(bio_model_path), BiorbdModel(bio_model_path))

    # Problem parameters
    n_shooting = (40, 40)
    final_time = (1, 1)
    tau_min, tau_max, tau_init = -300, 300, 0
    n_root = bio_model[0].nb_root
    n_q = bio_model[0].nb_q
    n_tau = n_q - n_root

    # Define the parameter to optimize
    parameters = ParameterList()

    parameters.add(
        "max_tau",  # The name of the parameter
        my_parameter_function,  # The function that modifies the biorbd model
        size=1,
    )

    parameter_bounds = BoundsList()
    parameter_init = InitialGuessList()

    parameter_init["max_tau"] = 0
    parameter_bounds.add("max_tau", min_bound=0, max_bound=tau_max, interpolation=InterpolationType.CONSTANT)

    # Add phase independant objective functions
    parameter_objectives = ParameterObjectiveList()
    parameter_objectives.add(ObjectiveFcn.Parameter.MINIMIZE_PARAMETER, key="max_tau", weight=1000, quadratic=True)

    # Add phase independant constraint functions
    parameter_constraints = ParameterConstraintList()
    parameter_constraints.add(ConstraintFcn.TRACK_PARAMETER, min_bound=-100, max_bound=100)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, phase=0, min_bound=0.1, max_bound=3)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, phase=1, min_bound=0.1, max_bound=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau_joints", weight=10, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau_joints", weight=10, phase=1)

    # Constraints
    constraints = ConstraintList()
    constraints.add(custom_constraint_parameters, node=Node.ALL, min_bound=0, max_bound=tau_max)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN_FREE_FLOATING_BASE, with_contact=False)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN_FREE_FLOATING_BASE, with_contact=False)

    # Path constraint
    x_bounds = BoundsList()
    q_roots_min = bio_model[0].bounds_from_ranges("q").min[:n_root, :]
    q_roots_max = bio_model[0].bounds_from_ranges("q").max[:n_root, :]
    q_joints_min = bio_model[0].bounds_from_ranges("q").min[n_root:, :]
    q_joints_max = bio_model[0].bounds_from_ranges("q").max[n_root:, :]
    qdot_roots_min = bio_model[0].bounds_from_ranges("qdot").min[:n_root, :]
    qdot_roots_max = bio_model[0].bounds_from_ranges("qdot").max[:n_root, :]
    qdot_joints_min = bio_model[0].bounds_from_ranges("qdot").min[n_root:, :]
    qdot_joints_max = bio_model[0].bounds_from_ranges("qdot").max[n_root:, :]
    x_bounds.add("q_roots", min_bound=q_roots_min, max_bound=q_roots_max, phase=0)
    x_bounds.add("q_joints", min_bound=q_joints_min, max_bound=q_joints_max, phase=0)
    x_bounds.add("qdot_roots", min_bound=qdot_roots_min, max_bound=qdot_roots_max, phase=0)
    x_bounds.add("qdot_joints", min_bound=qdot_joints_min, max_bound=qdot_joints_max, phase=0)
    x_bounds.add("q_roots", min_bound=q_roots_min, max_bound=q_roots_max, phase=1)
    x_bounds.add("q_joints", min_bound=q_joints_min, max_bound=q_joints_max, phase=1)
    x_bounds.add("qdot_roots", min_bound=qdot_roots_min, max_bound=qdot_roots_max, phase=1)
    x_bounds.add("qdot_joints", min_bound=qdot_joints_min, max_bound=qdot_joints_max, phase=1)

    # change model bound for -pi, pi
    for i in range(len(bio_model)):
        x_bounds[i]["q_joints"].min[0, :] = -np.pi
        x_bounds[i]["q_joints"].max[0, :] = np.pi

    # Phase 0
    x_bounds[0]["q_roots"][0, 0] = np.pi
    x_bounds[0]["q_joints"][0, 0] = 0
    x_bounds[0]["q_joints"].min[0, -1] = 6 * np.pi / 8 - 0.1
    x_bounds[0]["q_joints"].max[0, -1] = 6 * np.pi / 8 + 0.1

    # Phase 1
    x_bounds[1]["q_roots"][0, -1] = 3 * np.pi
    x_bounds[1]["q_joints"][0, -1] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(key="q_roots", initial_guess=[0] * n_root, phase=0)
    x_init.add(key="q_jointss", initial_guess=[0] * n_tau, phase=0)
    x_init.add(key="qdot_roots", initial_guess=[0] * n_root, phase=0)
    x_init.add(key="qdot_jointss", initial_guess=[0] * n_tau, phase=0)
    x_init.add(key="q_roots", initial_guess=[0] * n_root, phase=1)
    x_init.add(key="q_jointss", initial_guess=[0] * n_tau, phase=1)
    x_init.add(key="qdot_roots", initial_guess=[0] * n_root, phase=1)
    x_init.add(key="qdot_jointss", initial_guess=[0] * n_tau, phase=1)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(key="tau_joints", min_bound=[tau_min] * n_tau, max_bound=[tau_max] * n_tau, phase=0)
    u_bounds.add(key="tau_joints", min_bound=[tau_min] * n_tau, max_bound=[tau_max] * n_tau, phase=1)

    # Control initial guess
    u_init = InitialGuessList()
    u_init.add(key="tau_joints", initial_guess=[tau_init] * n_tau, phase=0)
    u_init.add(key="tau_joints", initial_guess=[tau_init] * n_tau, phase=1)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        parameter_objectives=parameter_objectives,
        parameter_constraints=parameter_constraints,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        constraints=constraints,
        parameters=parameters,
    )


def main():
    # --- Prepare the ocp --- #
    ocp = prepare_ocp()

    # --- Solve the ocp --- #
    sol = ocp.solve()

    # --- Show results --- #
    sol.animate()
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
