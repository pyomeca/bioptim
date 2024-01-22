"""
This example is inspired from the giant circle gymnastics skill. It is composed of two pendulums
representing the trunk and legs segments (only the hip flexion is actuated). The objective is to minimize the
maximum torque (minmax) of the hip flexion while performing the giant circle. The maximum torque is included to the
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
    ConstraintFcn,
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
)



def custom_constraint_max_tau(controller: PenaltyController) -> MX:
    return controller.parameters["max_tau"].cx - controller.controls["tau"].cx


def custom_constraint_min_tau(controller: PenaltyController) -> MX:
    return controller.parameters["min_tau"].cx - controller.controls["tau"].cx


def my_parameter_function(bio_model: biorbd.Model, value: MX):
    return


def prepare_ocp(
    bio_model_path: str = "models/double_pendulum.bioMod",
) -> OptimalControlProgram:
    bio_model = BiorbdModel(bio_model_path)

    # Problem parameters
    n_shooting = 50
    final_time = 1
    tau_min, tau_max, tau_init = -10, 10, 0

    # Mapping
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", to_second=[None, 0], to_first=[1])

    # Define the parameter to optimize
    parameters = ParameterList()
    parameter_init = InitialGuessList()
    parameter_bounds = BoundsList()

    parameters.add(
        "max_tau",
        my_parameter_function,
        size=1,
    )
    parameters.add(
        "min_tau",
        my_parameter_function,
        size=1,
    )

    parameter_init["max_tau"] = 1
    parameter_init["min_tau"] = -1

    parameter_bounds.add("max_tau", min_bound=0, max_bound=tau_max, interpolation=InterpolationType.CONSTANT)
    parameter_bounds.add("min_tau", min_bound=tau_min, max_bound=0, interpolation=InterpolationType.CONSTANT)

    # Add phase independent objective functions
    parameter_objectives = ParameterObjectiveList()
    parameter_objectives.add(ObjectiveFcn.Parameter.MINIMIZE_PARAMETER, key="max_tau", weight=10, quadratic=False)
    parameter_objectives.add(ObjectiveFcn.Parameter.MINIMIZE_PARAMETER, key="min_tau", weight=-10, quadratic=True)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, phase=0, min_bound=1, max_bound=5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)

    # Constraints
    constraints = ConstraintList()

    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, target=3.8)

    constraints.add(custom_constraint_max_tau, phase=0, node=Node.ALL_SHOOTING, min_bound=0, max_bound=tau_max)
    constraints.add(custom_constraint_min_tau, phase=0, node=Node.ALL_SHOOTING, min_bound=tau_min, max_bound=0)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(key="q", bounds=bio_model.bounds_from_ranges("q"))
    x_bounds.add(key="qdot", bounds=bio_model.bounds_from_ranges("qdot"))

    x_bounds["q"].min[0, :] = 0
    x_bounds["q"].max[0, :] = 3 * np.pi

    x_bounds["q"].min[1, :] = -np.pi / 3
    x_bounds["q"].max[1, :] = np.pi / 5

    x_bounds["q"][0, 0] = np.pi + 0.01
    x_bounds["q"][1, 0] = 0
    x_bounds["qdot"][0, 0] = 0
    x_bounds["qdot"][1, 0] = 0

    x_bounds["q"][0, -1] = 3 * np.pi
    x_bounds["q"][1, -1] = 0

    # Define control path constraint
    n_tau = len(tau_mappings[0]["tau"].to_first)
    u_bounds = BoundsList()
    u_bounds.add(key="tau", min_bound=[tau_min] * n_tau, max_bound=[tau_max] * n_tau, phase=0)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        parameter_objectives=parameter_objectives,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        constraints=constraints,
        variable_mappings=tau_mappings,
        parameters=parameters,
    )


def main():
    # --- Prepare the ocp --- #
    ocp = prepare_ocp()

    # --- Solve the ocp --- #
    sol = ocp.solve()
    # sol.print_cost()

    # --- Show results --- #
    # sol.animate()
#    sol.graphs(show_bounds=True)


    import matplotlib
    from matplotlib import pyplot as plt
    matplotlib.use('Qt5Agg')

    states = sol.decision_states()
    controls = sol.decision_controls()

    q = np.array([item.flatten() for item in states["q"]])
    qdot = np.array([item.flatten() for item in states["qdot"]])
    tau = np.vstack([
        np.array([item.flatten() for item in controls["tau"]]),
        np.array([[np.nan]])
    ])
    time = np.array([item.full().flatten()[0] for item in sol.stepwise_time()])


    fig, axs = plt.subplots(2, 2, figsize=(10, 15))

    # Plotting q solutions for both DOFs
    axs[0, 0].plot(time, q)
    axs[0, 0].set_title("Joint coordinates")
    axs[0, 0].set_ylabel("q")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].grid(True)

    axs[0, 1].plot(time, qdot)
    axs[0, 1].set_title("Joint velocities")
    axs[0, 1].set_ylabel("qdot")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].grid(True)

    axs[1, 0].step(time, tau)
    axs[1, 0].set_title("Generalized forces")
    axs[1, 0].set_ylabel("tau")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].grid(True)
    axs[1, 0].step(axs[1, 0].get_xlim(), np.ones([2]) * sol.parameters["max_tau"], 'k--')
    axs[1, 0].step(axs[1, 0].get_xlim(), np.ones([2]) * sol.parameters["min_tau"], 'k--')

    axs[1, 1].plot(sol.constraints[ocp.n_shooting * (ocp.nlp[0].model.nb_q + ocp.nlp[0].model.nb_qdot):], "o")
    axs[1, 1].set_title("Constaints (continuity excluded)")

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

    # Display the plot
    plt.show()

    print("Duration: ", time[-1])
    print('sum tau**2 dt = ', np.nansum(tau**2 * time[1]))
    print('min-max tau: ', np.nanmin(tau), np.nanmax(tau))


if __name__ == "__main__":
    main()

