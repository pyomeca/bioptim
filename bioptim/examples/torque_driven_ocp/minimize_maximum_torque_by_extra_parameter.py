import numpy as np
import biorbd_casadi as biorbd
from casadi import MX, mmax
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
    Bounds,
    InitialGuess,
    Objective,
    BiorbdModel,
    PenaltyController,
)


def custom_constraint_parameters(all_pn: PenaltyController) -> MX:
    tau = all_pn.nlp.controls["tau"].cx_start
    idx = all_pn.nlp.parameters.names.index("max_tau")
    max_param = all_pn.nlp.parameters[idx].cx
    val = max_param - tau
    return val


def minimize_parameters(ocp: OptimalControlProgram, value: MX) -> MX:
    return value


def my_parameter_function(bio_model: biorbd.Model, value: MX):
    return


def custom_min_parameter(controller: PenaltyController) -> MX:
    idx = controller.nlp.parameters.names.index("max_tau")
    # max_param = controller.nlp.parameters[idx].cx
    # val = controller.nlp.mx_to_cx(
    #     "min_max_tau", max_param, controller.nlp.parameters.cx_start
    # )
    return controller.nlp.parameters.cx_start[idx]  # val


def prepare_ocp(
    bio_model_path: str = "models/double_pendulum.bioMod",
) -> OptimalControlProgram:

    bio_model = (BiorbdModel(bio_model_path), BiorbdModel(bio_model_path))

    # Problem parameters
    n_shooting = (40, 40)
    final_time = (1, 1)
    tau_min, tau_max, tau_init = -300, 300, 0

    # Mapping
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", [None, 0], [1], phase=0)
    tau_mappings.add("tau", [None, 0], [1], phase=1)

    # Define the parameter to optimize
    parameters = ParameterList()

    parameter_initial_guess = InitialGuess(0)
    parameter_bounds = Bounds(0, tau_max, interpolation=InterpolationType.CONSTANT)

    parameter_objective_functions = Objective(minimize_parameters, weight=10000, quadratic=True, custom_type=ObjectiveFcn.Parameter)

    parameters.add(
        "max_tau",  # The name of the parameter
        my_parameter_function,  # The function that modifies the biorbd model
        parameter_initial_guess,  # The initial guess
        parameter_bounds,  # The bounds
        size=1,
        penalty_list=parameter_objective_functions,  # ObjectiveFcn of constraint for this particular parameter
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, phase=0, min_bound=0.1, max_bound=3)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, phase=1, min_bound=0.1, max_bound=3)
    objective_functions.add(custom_min_parameter, custom_type=ObjectiveFcn.Mayer, node=Node.ALL, weight=100000)

    # Constraints
    constraints = ConstraintList()
    constraints.add(custom_constraint_parameters, node=Node.ALL, min_bound=0, max_bound=tau_max)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    
    #change model bound for -pi, pi
    for i in range(len(bio_model)):
        x_bounds[i].min[1, :] = -np.pi
        x_bounds[i].max[1, :] = np.pi

    # Phase 0
    x_bounds[0][0, 0] = 3.14
    x_bounds[0][1, 0] = 0
    x_bounds[0].min[1, -1] = 6 * np.pi / 8 - 0.1
    x_bounds[0].max[1, -1] = 6 * np.pi / 8 + 0.1

    # Phase 1
    x_bounds[1][0, -1] = 3 * 3.14
    x_bounds[1][1, -1] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
    x_init.add([1] * (bio_model[1].nb_q + bio_model[1].nb_qdot))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * len(tau_mappings[0]["tau"].to_first), [tau_max] * len(tau_mappings[0]["tau"].to_first))
    u_bounds.add([tau_min] * len(tau_mappings[1]["tau"].to_first), [tau_max] * len(tau_mappings[1]["tau"].to_first))

    # Control initial guess
    u_init = InitialGuessList()
    u_init.add([tau_init] * len(tau_mappings[0]["tau"].to_first))
    u_init.add([tau_init] * len(tau_mappings[1]["tau"].to_first))

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
        constraints=constraints,
        variable_mappings=tau_mappings,
        parameters=parameters,
        assume_phase_dynamics=True
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
