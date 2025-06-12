from bioptim import (
    PhaseDynamics,
    SolutionMerge,
    TorqueBiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsOptionsList,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    ExternalForceSetTimeSeries,
)
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils


@pytest.mark.parametrize(
    "phase_dynamics",
    [
        PhaseDynamics.SHARED_DURING_THE_PHASE,
        PhaseDynamics.ONE_PER_NODE,
    ],
)
@pytest.mark.parametrize(
    "use_sx",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "method",
    [
        "translational_force",
        "in_global",
        "in_global_torque",
        "in_segment_torque",
        "in_segment",
        "translational_force_on_a_marker",
    ],
)
def test_example_external_forces(
    phase_dynamics,
    use_sx,
    method,
):
    from bioptim.examples.getting_started import example_external_forces as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_with_forces.bioMod",
        phase_dynamics=phase_dynamics,
        external_force_method=method,
        use_sx=use_sx,
        use_point_of_applications=method == "translational_force",  # Only to preserve the tested values
    )
    sol = ocp.solve()

    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], f[0, 0])

    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (246, 1))
    npt.assert_almost_equal(g, np.zeros((246, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    npt.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0, 0.0, 0.0]), decimal=5)
    npt.assert_almost_equal(qdot[:, -1], np.array([0.0, 0.0, 0.0, 0.0]), decimal=5)

    if method in ["in_global_torque", "in_segment_torque"]:
        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0, 0.0, 0.0]), decimal=5)
        npt.assert_almost_equal(qdot[:, -1], np.array([0.0, 0.0, 0.0, 0.0]), decimal=5)

        npt.assert_almost_equal(f[0, 0], 19847.887805189126)

        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array([2.03776708e-09, 1.27132259e01, 2.32230666e-27, 0.0]))
        npt.assert_almost_equal(tau[:, 10], np.array([-8.23139024e-10, 1.07110012e01, 5.45884198e-26, 0.0]))
        npt.assert_almost_equal(tau[:, 20], np.array([-6.72563418e-10, 8.70877651e00, -2.09699944e-26, 0.0]))
        npt.assert_almost_equal(tau[:, -1], np.array([2.03777147e-09, 6.90677426e00, -1.66232727e-25, 0.0]))

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array([-4.69166539e-15, 6.03679344e-16, -1.43453710e-17, 0.0]), decimal=5)
        npt.assert_almost_equal(q[:, -1], np.array([-4.69169398e-15, 2.00000000e00, -1.43453709e-17, 0.0]), decimal=5)

    if method in ["translational_force", "in_global", "in_segment"]:

        npt.assert_almost_equal(f[0, 0], 7067.851604540217)

        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array([2.03776698e-09, 6.98419368e00, -8.87085933e-09, 0.0]))
        npt.assert_almost_equal(tau[:, 10], np.array([-8.23139073e-10, 6.24337052e00, -9.08634557e-09, 0.0]))
        npt.assert_almost_equal(tau[:, 20], np.array([-6.72563331e-10, 5.50254736e00, -9.36203643e-09, 0.00]))
        npt.assert_almost_equal(tau[:, -1], np.array([2.03777157e-09, 4.83580652e00, -9.46966399e-09, 0.0]))

        # initial and final position

    if method == "translational_force":
        npt.assert_almost_equal(q[:, 0], np.array([-4.69166257e-15, 6.99774238e-16, -1.15797965e00, 0.0]), decimal=5)
        npt.assert_almost_equal(q[:, -1], np.array([-4.69169076e-15, 2.00000000e00, -1.15744926e00, 0.0]), decimal=5)

    if method == "in_global":
        npt.assert_almost_equal(q[:, 0], np.array([-3.41506199e-15, 6.99773953e-16, -4.11684795e-12, 0.0]), decimal=5)
        npt.assert_almost_equal(q[:, -1], np.array([-3.84536375e-15, 2.00000000e00, 6.27099006e-11, 0.0]), decimal=5)

    if method == "in_segment":
        npt.assert_almost_equal(q[:, 0], np.array([-4.64041726e-15, 6.99774094e-16, 4.75924448e-11, 0.0]), decimal=5)
        npt.assert_almost_equal(q[:, -1], np.array([-4.74298258e-15, 2.00000000e00, 3.79354612e-11, 0.0]), decimal=5)

    if method == "translational_force_on_a_marker":
        npt.assert_almost_equal(q[:, 0], np.array([-4.69167e-15, 7.80151e-16, -1.30653e-17, 0.00000e00]), decimal=5)
        npt.assert_almost_equal(q[:, -1], np.array([-4.69169e-15, 2.00000e00, -1.30653e-17, 0.00000e00]), decimal=5)
        npt.assert_almost_equal(qdot[:, 15], np.array([-3.39214e-19, 1.42151e00, -3.35346e-26, 0.00000e00]), decimal=5)


def prepare_ocp(
    biorbd_model_path: str = "models/cube_with_forces.bioMod",
    together: bool = False,
) -> OptimalControlProgram:

    # Problem parameters
    n_shooting = 30
    final_time = 2

    # Linear external forces
    Seg1_force = np.zeros((3, n_shooting))
    Seg1_force[2, :] = -2

    Test_force = np.zeros((3, n_shooting))
    Seg1_force[2, :] = -2
    Test_force[2, :] = 5
    Test_force[2, 4] = 52

    # Point of application
    Seg1_point_of_application = np.zeros((3, n_shooting))
    Seg1_point_of_application[0, :] = 0.05
    Seg1_point_of_application[1, :] = -0.05
    Seg1_point_of_application[2, :] = 0.007

    Test_point_of_application = np.zeros((3, n_shooting))
    Test_point_of_application[0, :] = -0.009
    Test_point_of_application[1, :] = 0.01
    Test_point_of_application[2, :] = -0.01

    # Torques external forces
    Seg1_torque = np.zeros((3, n_shooting))
    Seg1_torque[1, :] = 1.8
    Seg1_torque[1, 4] = 18

    Test_torque = np.zeros((3, n_shooting))
    Test_torque[1, :] = -1.8
    Test_torque[1, 4] = -18

    if together:
        external_forces = ExternalForceSetTimeSeries(nb_frames=n_shooting)
        external_forces.add("force0", "Seg1", np.vstack((Seg1_torque, Seg1_force)), Seg1_point_of_application)
        external_forces.add("force1", "Test", np.vstack((Test_torque, Test_force)), Test_point_of_application)

    else:
        external_forces = ExternalForceSetTimeSeries(nb_frames=n_shooting)
        external_forces.add(
            "force2", "Seg1", np.vstack((Seg1_torque, np.zeros((3, n_shooting)))), point_of_application=Seg1_point_of_application
        )
        external_forces.add(
            "force3", "Seg1", np.vstack((np.zeros((3, n_shooting)), Seg1_force)), point_of_application=Seg1_point_of_application
        )
        external_forces.add(
            "force4", "Test", np.vstack((Test_torque, np.zeros((3, n_shooting)))), point_of_application=Test_point_of_application
        )
        external_forces.add(
            "force5", "Test", np.vstack((np.zeros((3, n_shooting)), Test_force)), point_of_application=Test_point_of_application
        )

    bio_model = TorqueBiorbdModel(biorbd_model_path, external_force_set=external_forces)
    numerical_data_timeseries = {"external_forces": external_forces.to_numerical_time_series()}

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(
        numerical_data_timeseries=numerical_data_timeseries,
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][3, [0, -1]] = 0
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:3, [0, -1]] = 0

    # Define control path constraint
    u_bounds = BoundsList()
    tau_min, tau_max = -100, 100
    u_bounds["tau"] = [tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
    )


@pytest.mark.parametrize(
    "together",
    [
        True,
        False,
    ],
)
def test_example_external_forces_all_at_once(together: bool):

    from bioptim.examples.getting_started import example_external_forces as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_with_forces.bioMod",
        together=together,
    )

    sol = ocp.solve()
    # # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 5507.938264053537)
    npt.assert_almost_equal(f[0, 0], sol.detailed_cost[0]["cost_value_weighted"])

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (246, 1))
    npt.assert_almost_equal(g, np.zeros((246, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array([-0.17782387, 4.9626883, -0.11993298, 0.0]))
    npt.assert_almost_equal(tau[:, 10], np.array([0.0590585, 5.15623667, -0.11993298, 0.0]))
    npt.assert_almost_equal(tau[:, 20], np.array([0.03581958, 5.34978503, -0.11993298, 0.0]))
    npt.assert_almost_equal(tau[:, -1], np.array([-0.12181904, 5.52397856, -0.11993298, 0.0]))

    # initial and final position
    npt.assert_almost_equal(
        q[:, 0], np.array([-3.05006547e-15, 7.80152284e-16, -5.64943197e-03, 0.00000000e00]), decimal=5
    )
    npt.assert_almost_equal(
        q[:, -1], np.array([-3.85629113e-15, 2.00000000e00, -5.02503164e-04, 0.00000000e00]), decimal=5
    )

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0, 0.0, 0.0]), decimal=5)
    npt.assert_almost_equal(qdot[:, -1], np.array([0.0, 0.0, 0.0, 0.0]), decimal=5)

    # how the force is stored
    data_Seg1 = ocp.nlp[0].numerical_data_timeseries["external_forces"]
    if together:
        npt.assert_equal(data_Seg1.shape, (18, 1, 31))
        no_zeros_data_Seg1 = data_Seg1[~np.all(np.all(data_Seg1 == 0, axis=1), axis=1)]
        npt.assert_equal(no_zeros_data_Seg1.shape, (10, 1, 31))
        npt.assert_almost_equal(
            no_zeros_data_Seg1[:, 0, 4],
            np.array([1.8e01, -2.0e00, 5.0e-02, -5.0e-02, 7.0e-03, -1.8e01, 5.2e01, -9.0e-03, 1.0e-02, -1.0e-02]),
        )
    else:
        npt.assert_equal(data_Seg1.shape, (36, 1, 31))
        no_zeros_data_Seg1 = data_Seg1[~np.all(np.all(data_Seg1 == 0, axis=1), axis=1)]
        npt.assert_equal(no_zeros_data_Seg1.shape, (16, 1, 31))
        npt.assert_almost_equal(
            no_zeros_data_Seg1[:, 0, 4],
            np.array(
                [
                    1.8e01,
                    5.0e-02,
                    -5.0e-02,
                    7.0e-03,
                    -2.0e00,
                    5.0e-02,
                    -5.0e-02,
                    7.0e-03,
                    -1.8e01,
                    -9.0e-03,
                    1.0e-02,
                    -1.0e-02,
                    5.2e01,
                    -9.0e-03,
                    1.0e-02,
                    -1.0e-02,
                ]
            ),
        )
