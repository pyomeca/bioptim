import numpy as np
from casadi import MX
import pytest

from bioptim import (
    BiorbdModel,
    BoundsList,
    Bounds,
    ConstraintFcn,
    ConstraintList,
    DynamicsFcn,
    DynamicsList,
    InitialGuessList,
    InitialGuess,
    Node,
    InterpolationType,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    ParameterObjectiveList,
    PenaltyController,
)
from bioptim.optimization.solution import Solution

from .utils import TestUtils


def prepare_ocp(phase_time_constraint, use_parameter, assume_phase_dynamics):
    # --- Inputs --- #
    final_time = (2, 5, 4)
    time_min = [1, 3, 0.1]
    time_max = [2, 4, 0.8]
    ns = (20, 30, 20)
    biorbd_model_path = TestUtils.bioptim_folder() + "/examples/optimal_time_ocp/models/cube.bioMod"
    ode_solver = OdeSolver.RK4()

    # --- Options --- #
    n_phases = len(ns)

    # BioModel path
    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path))

    # Problem parameters
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m1", phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=2)

    constraints.add(
        ConstraintFcn.TIME_CONSTRAINT,
        node=Node.END,
        minimum=time_min[0],
        maximum=time_max[0],
        phase=phase_time_constraint,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))  # Phase 0
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))  # Phase 1
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))  # Phase 2

    for bounds in x_bounds:
        for i in [1, 3, 4, 5]:
            bounds[i, [0, -1]] = 0
    x_bounds[0][2, 0] = 0.0
    x_bounds[2][2, [0, -1]] = [0.0, 1.57]

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model[0].nb_tau)
    u_init.add([tau_init] * bio_model[0].nb_tau)
    u_init.add([tau_init] * bio_model[0].nb_tau)

    parameters = ParameterList()
    parameter_objectives = ParameterObjectiveList()
    if use_parameter:

        def my_target_function(controller: PenaltyController):
            return controller.parameters.cx

        def my_parameter_function(bio_model, value, extra_value):
            new_gravity = MX.zeros(3, 1)
            new_gravity[2] = value + extra_value
            bio_model.set_gravity(new_gravity)

        min_g = -10
        max_g = -6
        target_g = -8
        bound_gravity = Bounds(min_g, max_g, interpolation=InterpolationType.CONSTANT)
        initial_gravity = InitialGuess((min_g + max_g) / 2)
        parameters.add(
            "gravity_z",
            my_parameter_function,
            initial_gravity,
            bound_gravity,
            size=1,
            extra_value=1,
        )
        parameter_objectives.add(my_target_function,  weight=10, quadratic=True, custom_type=ObjectiveFcn.Parameter, target=target_g)

    # ------------- #

    return OptimalControlProgram(
        bio_model[:n_phases],
        dynamics,
        ns,
        final_time[:n_phases],
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
        parameters=parameters,
        parameter_objectives=parameter_objectives,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("phase_time_constraint", [0, 1, 2])
@pytest.mark.parametrize("use_parameter", [True, True])
def test_variable_time(phase_time_constraint, use_parameter, assume_phase_dynamics):
    ocp = prepare_ocp(phase_time_constraint, use_parameter, assume_phase_dynamics)

    # --- Solve the program --- #
    np.random.seed(42)
    sol = Solution(ocp, np.random.random((649 + use_parameter, 1)))

    # --- Show results --- #
    states, controls, parameters = sol.states, sol.controls, sol.parameters

    np.testing.assert_almost_equal(
        states[0]["q"][0, 0:8],
        np.array([0.37454012, 0.05808361, 0.83244264, 0.43194502, 0.45606998, 0.60754485, 0.30461377, 0.03438852]),
    )
    np.testing.assert_almost_equal(
        states[1]["q"][0, 0:8],
        np.array([0.81801477, 0.11986537, 0.3636296, 0.28484049, 0.90826589, 0.67213555, 0.63352971, 0.04077514]),
    )
    np.testing.assert_almost_equal(
        states[2]["q"][0, 0:8],
        np.array([0.02535074, 0.15643704, 0.95486528, 0.35597268, 0.85546058, 0.17320187, 0.37461261, 0.07056875]),
    )

    np.testing.assert_almost_equal(
        states[0]["qdot"][0, 0:8],
        np.array([0.59865848, 0.70807258, 0.18340451, 0.13949386, 0.51423444, 0.94888554, 0.44015249, 0.66252228]),
    )
    np.testing.assert_almost_equal(
        states[1]["qdot"][0, 0:8],
        np.array([0.5107473, 0.32320293, 0.2517823, 0.50267902, 0.48945276, 0.72821635, 0.8353025, 0.01658783]),
    )
    np.testing.assert_almost_equal(
        states[2]["qdot"][0, 0:8],
        np.array([0.69597421, 0.71459592, 0.61172075, 0.11607264, 0.09783416, 0.6158501, 0.85648984, 0.58577558]),
    )

    np.testing.assert_almost_equal(
        controls[0]["tau"][0, 0:8],
        np.array([0.70624223, 0.98663958, 0.81279957, 0.75337819, 0.77714692, 0.90635439, 0.01135364, 0.11881792]),
    )
    np.testing.assert_almost_equal(
        controls[1]["tau"][0, 0:8],
        np.array([0.97439481, 0.53609637, 0.68473117, 0.82253724, 0.6134152, 0.86606389, 0.37646337, 0.15041689]),
    )
    np.testing.assert_almost_equal(
        controls[2]["tau"][0, 0:8],
        np.array([0.12804584, 0.64087474, 0.89678841, 0.17231987, 0.16893506, 0.08870253, 0.20633372, 0.69039483]),
    )

    np.testing.assert_almost_equal(
        controls[2]["tau"][0, 0:8],
        np.array([0.12804584, 0.64087474, 0.89678841, 0.17231987, 0.16893506, 0.08870253, 0.20633372, 0.69039483]),
    )

    np.testing.assert_almost_equal(parameters["gravity_z"], 0.78917124)
    np.testing.assert_almost_equal(parameters["time"], 0.4984422)
