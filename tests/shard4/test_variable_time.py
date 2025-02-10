import numpy as np
import numpy.testing as npt
import pytest
from casadi import MX

from bioptim import (
    BiorbdModel,
    BoundsList,
    ConstraintFcn,
    ConstraintList,
    DynamicsFcn,
    DynamicsList,
    InitialGuessList,
    Node,
    InterpolationType,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    ParameterObjectiveList,
    PenaltyController,
    PhaseDynamics,
    SolutionMerge,
    VariableScaling,
)
from bioptim.optimization.solution.solution import Solution
from tests.utils import TestUtils


def prepare_ocp(phase_time_constraint, use_parameter, phase_dynamics):
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
    tau_min, tau_max = -100, 100

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0, expand_dynamics=True, phase_dynamics=phase_dynamics)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=1, expand_dynamics=True, phase_dynamics=phase_dynamics)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2, expand_dynamics=True, phase_dynamics=phase_dynamics)

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
    x_bounds.add("q", bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds.add("q", bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds.add("q", bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bio_model[2].bounds_from_ranges("qdot"), phase=2)
    assert x_bounds.nb_phase == n_phases

    for bounds in x_bounds:
        bounds["q"][1, [0, -1]] = 0
        bounds["qdot"][:, [0, -1]] = 0
    x_bounds[0]["q"][2, 0] = 0.0
    x_bounds[2]["q"][2, [0, -1]] = [0.0, 1.57]

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=0)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[1].nb_tau, max_bound=[tau_max] * bio_model[1].nb_tau, phase=1)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[2].nb_tau, max_bound=[tau_max] * bio_model[2].nb_tau, phase=2)

    parameters = ParameterList(use_sx=False)
    parameter_bounds = BoundsList()
    parameter_init = InitialGuessList()
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
        parameters.add(
            "gravity_z", my_parameter_function, size=1, extra_value=1, scaling=VariableScaling("gravity_z", [1])
        )
        parameter_objectives.add(
            my_target_function, weight=10, quadratic=True, custom_type=ObjectiveFcn.Parameter, target=target_g
        )
        parameter_bounds.add(
            "gravity_z", min_bound=[min_g], max_bound=[max_g], interpolation=InterpolationType.CONSTANT
        )
        parameter_init["gravity_z"] = (min_g + max_g) / 2

    # ------------- #

    return OptimalControlProgram(
        bio_model[:n_phases],
        dynamics,
        ns,
        final_time[:n_phases],
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        parameters=parameters,
        parameter_init=parameter_init,
        parameter_bounds=parameter_bounds,
        parameter_objectives=parameter_objectives,
    )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("phase_time_constraint", [0, 1, 2])
@pytest.mark.parametrize("use_parameter", [True, True])
def test_variable_time(phase_time_constraint, use_parameter, phase_dynamics):
    ocp = prepare_ocp(phase_time_constraint, use_parameter, phase_dynamics)

    # --- Solve the program --- #
    np.random.seed(42)
    time_init = np.array([1.23, 4.56, 7.89])[:, np.newaxis]
    sol = Solution.from_vector(ocp, np.concatenate((time_init, np.random.random((649 + use_parameter, 1)))))

    # --- Show results --- #
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(
        states[0]["q"][0, 0:8],
        np.array([0.37454012, 0.05808361, 0.83244264, 0.43194502, 0.45606998, 0.60754485, 0.30461377, 0.03438852]),
    )
    npt.assert_almost_equal(
        states[1]["q"][0, 0:8],
        np.array([0.81801477, 0.11986537, 0.3636296, 0.28484049, 0.90826589, 0.67213555, 0.63352971, 0.04077514]),
    )
    npt.assert_almost_equal(
        states[2]["q"][0, 0:8],
        np.array([0.02535074, 0.15643704, 0.95486528, 0.35597268, 0.85546058, 0.17320187, 0.37461261, 0.07056875]),
    )

    npt.assert_almost_equal(
        states[0]["qdot"][0, 0:8],
        np.array([0.59865848, 0.70807258, 0.18340451, 0.13949386, 0.51423444, 0.94888554, 0.44015249, 0.66252228]),
    )
    npt.assert_almost_equal(
        states[1]["qdot"][0, 0:8],
        np.array([0.5107473, 0.32320293, 0.2517823, 0.50267902, 0.48945276, 0.72821635, 0.8353025, 0.01658783]),
    )
    npt.assert_almost_equal(
        states[2]["qdot"][0, 0:8],
        np.array([0.69597421, 0.71459592, 0.61172075, 0.11607264, 0.09783416, 0.6158501, 0.85648984, 0.58577558]),
    )

    npt.assert_almost_equal(
        controls[0]["tau"][0, 0:8],
        np.array([0.70624223, 0.98663958, 0.81279957, 0.75337819, 0.77714692, 0.90635439, 0.01135364, 0.11881792]),
    )
    npt.assert_almost_equal(
        controls[1]["tau"][0, 0:8],
        np.array([0.97439481, 0.53609637, 0.68473117, 0.82253724, 0.6134152, 0.86606389, 0.37646337, 0.15041689]),
    )
    npt.assert_almost_equal(
        controls[2]["tau"][0, 0:8],
        np.array([0.12804584, 0.64087474, 0.89678841, 0.17231987, 0.16893506, 0.08870253, 0.20633372, 0.69039483]),
    )

    npt.assert_almost_equal(
        controls[2]["tau"][0, 0:8],
        np.array([0.12804584, 0.64087474, 0.89678841, 0.17231987, 0.16893506, 0.08870253, 0.20633372, 0.69039483]),
    )

    npt.assert_almost_equal(sol.parameters["gravity_z"], 0.78917124)
