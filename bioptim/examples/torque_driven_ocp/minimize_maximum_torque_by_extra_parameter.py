"""
This example is inspired from the giant circle gymnastics skill. It is composed of two pendulums
representing the trunk and legs segments (only the hip flexion is actuated). The objective is to minimize the
maximum torque (minmax) of the hip flexion while performing the giant circle. The maximum torque is included to the
problem as a parameter, all torque interval re constrained to be smaller than this parameter, this parameter is the
minimized. Two options are provided and compared to define initial and final states (0: bounds; 1: constraints)
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
    VariableScaling,
)
from matplotlib import pyplot as plt


def custom_constraint_max_tau(controller: PenaltyController) -> MX:
    return controller.parameters["max_tau"].cx - controller.controls["tau"].cx


def custom_constraint_min_tau(controller: PenaltyController) -> MX:
    return controller.parameters["min_tau"].cx - controller.controls["tau"].cx


def my_parameter_function(bio_model: biorbd.Model, value: MX):
    return


def prepare_ocp(
    method_bound_states: int = 0,
    bio_model_path: str = "models/double_pendulum.bioMod",
) -> OptimalControlProgram:

    # Problem parameters
    n_shooting = 50
    final_time = 1
    tau_min, tau_max, tau_init = -10, 10, 0

    # Define the parameter to optimize
    parameters = ParameterList(use_sx=False)
    parameter_init = InitialGuessList()
    parameter_bounds = BoundsList()

    parameters.add("max_tau", my_parameter_function, size=1, scaling=VariableScaling("max_tau", [1]))
    parameters.add("min_tau", my_parameter_function, size=1, scaling=VariableScaling("min_tau", [1]))

    parameter_init["max_tau"] = 1
    parameter_init["min_tau"] = -1

    parameter_bounds.add("max_tau", min_bound=0, max_bound=tau_max, interpolation=InterpolationType.CONSTANT)
    parameter_bounds.add("min_tau", min_bound=tau_min, max_bound=0, interpolation=InterpolationType.CONSTANT)

    # Add phase independent objective functions
    parameter_objectives = ParameterObjectiveList()
    parameter_objectives.add(ObjectiveFcn.Parameter.MINIMIZE_PARAMETER, key="max_tau", weight=10, quadratic=True)
    parameter_objectives.add(ObjectiveFcn.Parameter.MINIMIZE_PARAMETER, key="min_tau", weight=10, quadratic=True)

    # Define the model
    bio_model = BiorbdModel(bio_model_path, parameters=parameters)

    # Mapping
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", to_second=[None, 0], to_first=[1])

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
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(key="q", bounds=bio_model.bounds_from_ranges("q"))
    x_bounds.add(key="qdot", bounds=bio_model.bounds_from_ranges("qdot"))

    x_bounds["q"].min[0, :] = 0
    x_bounds["q"].max[0, :] = 3 * np.pi

    x_bounds["q"].min[1, :] = -np.pi / 3
    x_bounds["q"].max[1, :] = np.pi / 5

    if method_bound_states == 0:
        x_bounds["q"][0, 0] = np.pi + 0.01
        x_bounds["q"][1, 0] = 0
        x_bounds["qdot"][0, 0] = 0
        x_bounds["qdot"][1, 0] = 0

        x_bounds["q"][0, -1] = 3 * np.pi
        x_bounds["q"][1, -1] = 0
    else:
        constraints.add(ConstraintFcn.TRACK_STATE, node=Node.START, key="q", target=[np.pi + 0.01, 0])
        constraints.add(ConstraintFcn.TRACK_STATE, node=Node.START, key="qdot", target=[0, 0])
        constraints.add(ConstraintFcn.TRACK_STATE, node=Node.END, key="q", target=[3 * np.pi, 0])

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        objective_functions=objective_functions,
        parameter_objectives=parameter_objectives,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        constraints=constraints,
        parameters=parameters,
    )


def main():
    # --- Prepare the ocp --- #
    fig, axs = plt.subplots(1, 3)
    axs[0].set_title("Joint coordinates")
    axs[0].set_ylabel("q [°]")
    axs[1].set_title("Joint velocities")
    axs[1].set_ylabel("qdot [°/s]")
    axs[2].set_title("Generalized forces")
    axs[2].set_ylabel("tau [Nm]")
    for ax in [0, 1, 2]:
        axs[ax].set_xlabel("Time [s]")
        axs[ax].grid(True)

    linestyles = ["solid", "dashed"]

    for i, linestyle in enumerate(linestyles):
        # --- Prepare the ocp --- #
        ocp = prepare_ocp(method_bound_states=i)

        # --- Solve the ocp --- #
        sol = ocp.solve()
        # sol.print_cost()

        # --- Show results --- #
        # sol.animate()
        #    sol.graphs(show_bounds=True)

        states = sol.decision_states()
        controls = sol.decision_controls()

        q = np.array([item.flatten() for item in states["q"]])
        qdot = np.array([item.flatten() for item in states["qdot"]])
        tau = np.vstack([np.array([item.flatten() for item in controls["tau"]]), np.array([[np.nan]])])
        time = np.array([item.full().flatten()[0] for item in sol.stepwise_time()])

        print("Duration: ", time[-1])
        print("sum tau**2 dt = ", np.nansum(tau**2 * time[1]))
        print("min-max tau: ", np.nanmin(tau), np.nanmax(tau))

        # Plotting q solutions for both DOFs
        axs[0].plot(
            time,
            q * 180 / np.pi,
            linestyle=linestyle,
            label=[f"q_0 (method {i})", f"q_1 (method {i})"],
        )
        axs[1].plot(
            time,
            qdot * 180 / np.pi,
            linestyle=linestyle,
            label=[f"qdot_0 (method {i})", f"qdot_1 (method {i})"],
        )
        axs[2].step(time, tau, linestyle=linestyle, label=f"tau (method {i})", where="post")
        xlim = np.asarray(axs[2].get_xlim())
        xlim = np.hstack([xlim, np.nan, xlim])
        min_max = np.hstack(
            [np.ones([2]) * sol.parameters["min_tau"], np.nan, np.ones([2]) * sol.parameters["max_tau"]]
        )
        axs[2].plot(xlim, min_max, linestyle=(0, (5, 10)), label=f"params - bounds (method {i})")

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

    # Display the plot
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.show()


if __name__ == "__main__":
    main()
