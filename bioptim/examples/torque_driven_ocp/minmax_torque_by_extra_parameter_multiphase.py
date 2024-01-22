"""
This example is inspired from the clear pike circle gymnastics skill. It is composed of two pendulums
representing the trunk and legs segments (only the hip flexion is actuated). The objective is to minimize the
extreme torque (minmax) of the hip flexion while performing the clear pike circle motion. The extreme torques are included to the
problem as parameters, all torque intervals are constrained to be smaller than this parameter, these parameters are
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

def custom_constraint_min_max_tau(controller: PenaltyController) -> MX:
    mini = controller.parameters["min_max_tau"].cx[0] - controller.controls["tau"].cx
    maxi = controller.parameters["min_max_tau"].cx[1] - controller.controls["tau"].cx
    return controller.parameters["min_max_tau"].cx - controller.controls["tau"].cx#[mini, maxi]


def my_parameter_function(bio_model: biorbd.Model, value: MX):
    return


def prepare_ocp(
    bio_model_path: str = "models/double_pendulum.bioMod",
) -> OptimalControlProgram:
    bio_model = (BiorbdModel(bio_model_path), BiorbdModel(bio_model_path))

    # Problem parameters
    n_shooting = (30, 30)
    final_time = (2, 3)
    tau_min, tau_max, tau_init = -40, 40, 0

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Mapping
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", to_second=[None, 0], to_first=[1])
    tau_mappings.add("tau", to_second=[None, 0], to_first=[1])


    constraints = ConstraintList()

    # Define the parameter to optimize
    parameters = ParameterList()
    parameter_init = InitialGuessList()
    parameter_bounds = BoundsList()
    parameter_objectives = ParameterObjectiveList()

    method = 1

    if method == 1:
        parameters.add("max_tau", my_parameter_function, size=1, )
        parameters.add("min_tau", my_parameter_function, size=1, )
        parameters.add("to_test", my_parameter_function, size=1, )


        parameter_init["max_tau"] = 1
        parameter_init["min_tau"] = -1
        parameter_init["to_test"] = 1000


        parameter_bounds.add("max_tau", min_bound=0, max_bound=tau_max, interpolation=InterpolationType.CONSTANT)
        parameter_bounds.add("min_tau", min_bound=tau_min, max_bound=0, interpolation=InterpolationType.CONSTANT)
        parameter_bounds.add("to_test", min_bound=100, max_bound=1000, interpolation=InterpolationType.CONSTANT)

        # Add phase independent objective functions

        parameter_objectives.add(ObjectiveFcn.Parameter.MINIMIZE_PARAMETER, key="max_tau", weight=10, quadratic=True)
        parameter_objectives.add(ObjectiveFcn.Parameter.MINIMIZE_PARAMETER, key="min_tau", weight=10, quadratic=True)

        # Constraints
        constraints.add(custom_constraint_max_tau, phase=0, node=Node.ALL_SHOOTING, min_bound=0, max_bound=tau_max)
        constraints.add(custom_constraint_min_tau, phase=0, node=Node.ALL_SHOOTING, min_bound=tau_min, max_bound=0)
        #
        constraints.add(custom_constraint_max_tau, phase=1, node=Node.ALL_SHOOTING, min_bound=0, max_bound=tau_max)
        constraints.add(custom_constraint_min_tau, phase=1, node=Node.ALL_SHOOTING, min_bound=tau_min, max_bound=0)

    elif method == 2: #todo: does not work
        parameters.add("min_max_tau", my_parameter_function, size=2, )
        parameter_init["min_max_tau"] = [-1, 1]
        parameter_bounds.add("min_max_tau",
                             min_bound=[tau_min, 0],
                             max_bound=[0, tau_max],
                             interpolation=InterpolationType.CONSTANT
                             )

        # Add phase independent objective functions
        parameter_objectives.add(ObjectiveFcn.Parameter.MINIMIZE_PARAMETER, key="min_max_tau", weight=10, quadratic=True)

        # Constraints

        for phase in range(len(n_shooting)):
            constraints.add(
                custom_constraint_min_max_tau,
                phase=phase, node=Node.ALL_SHOOTING,
                min_bound=[tau_min, 0], max_bound=[0, tau_max]
            )

    objective_functions = ObjectiveList()
    # Add objective functions
    for phase in range(len(n_shooting)):
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1e-5, phase=phase, min_bound=0.5, max_bound=3) #was w=10
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1e-5, phase=phase) #was w=1



    constraints.add(
        ConstraintFcn.TRACK_STATE, key="q", phase=0, node=Node.START, target=[np.pi + 0.01, 0]
    )  # Initial state
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", phase=0, node=Node.START, target=[0, 0])
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", index=1, phase=1, node=Node.START, target=np.pi)
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", phase=1, node=Node.END, target=[3 * np.pi, 0])  # Final state

    for i in range(len(bio_model)):
        constraints.add(
            ConstraintFcn.BOUND_STATE,
            key="q",
            phase=i,
            node=Node.ALL,
            min_bound=[np.pi, -np.pi / 3],
            max_bound=[3 * np.pi, np.pi],
        ),

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
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

    print(sol.parameters)
    # sol.print_cost()

    import matplotlib
    from matplotlib import pyplot as plt
    matplotlib.use('Qt5Agg')

    states = sol.decision_states()
    controls = sol.decision_controls()

    q = np.vstack([
        np.array([item.flatten() for item in states[0]["q"]]),
        np.array([item.flatten() for item in states[1]["q"]])
    ])
    qdot = np.vstack([
        np.array([item.flatten() for item in states[0]["qdot"]]),
        np.array([item.flatten() for item in states[1]["qdot"]])
    ])
    tau = np.vstack([
        np.array([item.flatten() for item in controls[0]["tau"]]),
        np.array([[np.nan]]),
        np.array([item.flatten() for item in controls[1]["tau"]]),
        np.array([[np.nan]]),
    ])
    time = np.vstack([
        np.array([item.full().flatten()[0] for item in sol.stepwise_time()[0]]).reshape((-1, 1)),
        np.array([item.full().flatten()[0] for item in sol.stepwise_time()[1]]).reshape((-1, 1)) + sol.stepwise_time()[0][-1].full(),
    ])

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

    if len(sol.parameters) == 1:
        for i in range(2):
            axs[1, 0].step(axs[1, 0].get_xlim(), np.ones([2]) * sol.parameters["min_max_tau"][i], 'k--')
    else:
        axs[1, 0].step(axs[1, 0].get_xlim(), np.ones([2]) * sol.parameters["max_tau"], 'k--')
        axs[1, 0].step(axs[1, 0].get_xlim(), np.ones([2]) * sol.parameters["min_tau"], 'k--')

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

    # Display the plot
    plt.show()



    # --- Show results --- #
    sol.animate()
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()

