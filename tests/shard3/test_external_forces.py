import os

import numpy as np
import numpy.testing as npt
import pytest

from bioptim import (
    PhaseDynamics,
    SolutionMerge,
    ExternalForceType,
    ReferenceFrame,
    BiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ExternalForces,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("use_sx", [True, False])
@pytest.mark.parametrize("force_type", [ExternalForceType.FORCE, ExternalForceType.TORQUE])
@pytest.mark.parametrize("force_reference_frame", [ReferenceFrame.GLOBAL, ReferenceFrame.LOCAL])
@pytest.mark.parametrize("use_point_of_applications", [True, False])
@pytest.mark.parametrize("point_of_application_reference_frame", [ReferenceFrame.GLOBAL, ReferenceFrame.LOCAL])
def test_example_external_forces(phase_dynamics,
                                 use_sx,
                                 force_type,
                                 force_reference_frame,
                                 use_point_of_applications,
                                 point_of_application_reference_frame):
    from bioptim.examples.getting_started import example_external_forces as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    if use_point_of_applications == False:
        point_of_application_reference_frame = None

    if force_type == ExternalForceType.FORCE and force_reference_frame == ReferenceFrame.GLOBAL and point_of_application_reference_frame == ReferenceFrame.GLOBAL:
        # This combination is already tested in test_global_getting_started.py
        return

    # Errors for some combinations
    if force_type == ExternalForceType.TORQUE and use_point_of_applications:
        with pytest.raises(
                ValueError,
                match="Point of application cannot be used with ExternalForceType.TORQUE",
        ):
            ocp = ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/cube_with_forces.bioMod",
                phase_dynamics=phase_dynamics,
                force_type=force_type,
                force_reference_frame=force_reference_frame,
                use_point_of_applications=use_point_of_applications,
                point_of_application_reference_frame=point_of_application_reference_frame,
                use_sx=use_sx,
            )
        return

    if force_reference_frame == ReferenceFrame.LOCAL and point_of_application_reference_frame == ReferenceFrame.GLOBAL:
        with pytest.raises(
                NotImplementedError,
                match="External forces in local reference frame cannot have a point of application in the global reference frame yet",
        ):
            ocp = ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/cube_with_forces.bioMod",
                phase_dynamics=phase_dynamics,
                force_type=force_type,
                force_reference_frame=force_reference_frame,
                use_point_of_applications=use_point_of_applications,
                point_of_application_reference_frame=point_of_application_reference_frame,
                use_sx=use_sx,
            )
        return

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_with_forces.bioMod",
        phase_dynamics=phase_dynamics,
        force_type=force_type,
        force_reference_frame=force_reference_frame,
        use_point_of_applications=use_point_of_applications,
        point_of_application_reference_frame=point_of_application_reference_frame,
        use_sx=use_sx,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (246, 1))
    npt.assert_almost_equal(g, np.zeros((246, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0, 0.0, 0.0]), decimal=5)
    npt.assert_almost_equal(qdot[:, -1], np.array([0.0, 0.0, 0.0, 0.0]), decimal=5)

    if force_type == ExternalForceType.TORQUE:

        npt.assert_almost_equal(f[0, 0], 19847.887805189126)

        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array([2.03776708e-09, 1.27132259e+01, 2.32230666e-27, 0.0]))
        npt.assert_almost_equal(tau[:, 10], np.array([-8.23139024e-10,  1.07110012e+01,  5.45884198e-26,  0.0]))
        npt.assert_almost_equal(tau[:, 20], np.array([-6.72563418e-10,  8.70877651e+00, -2.09699944e-26,  0.0]))
        npt.assert_almost_equal(tau[:, -1], np.array([2.03777147e-09,  6.90677426e+00, -1.66232727e-25,  0.0]))

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array([-4.69166539e-15,  6.03679344e-16, -1.43453710e-17,  0.0]), decimal=5)
        npt.assert_almost_equal(q[:, -1], np.array([-4.69169398e-15,  2.00000000e+00, -1.43453709e-17,  0.0]), decimal=5)

        # how the force is stored
        if force_reference_frame == ReferenceFrame.LOCAL:
            data = ocp.nlp[0].numerical_data_timeseries["forces_in_local"]
        else:
            data = ocp.nlp[0].numerical_data_timeseries["forces_in_global"]
        npt.assert_equal(data.shape, (9, 2, 31))
        npt.assert_almost_equal(data[:, 0, 0], np.array([0., 0., -2., 0., 0., 0., 0., 0., 0.]))

    else:
        if force_reference_frame == ReferenceFrame.GLOBAL:
            if use_point_of_applications:
                if point_of_application_reference_frame == ReferenceFrame.LOCAL:
                    npt.assert_almost_equal(f[0, 0], 7067.851604540217)

                    # initial and final controls
                    npt.assert_almost_equal(tau[:, 0], np.array([2.03776698e-09,  6.98419368e+00, -8.87085933e-09,  0.0]))
                    npt.assert_almost_equal(tau[:, 10], np.array([-8.23139073e-10,  6.24337052e+00, -9.08634557e-09,  0.0]))
                    npt.assert_almost_equal(tau[:, 20], np.array([-6.72563331e-10,  5.50254736e+00, -9.36203643e-09,  0.00]))
                    npt.assert_almost_equal(tau[:, -1], np.array([2.03777157e-09,  4.83580652e+00, -9.46966399e-09,  0.0]))

                    # initial and final position
                    npt.assert_almost_equal(q[:, 0], np.array([-4.69166257e-15,  6.99774238e-16, -1.15797965e+00,  0.0]),
                                            decimal=5)
                    npt.assert_almost_equal(q[:, -1], np.array([-4.69169076e-15,  2.00000000e+00, -1.15744926e+00,  0.0]),
                                            decimal=5)

                    # how the force is stored
                    data = ocp.nlp[0].numerical_data_timeseries["translational_forces"]
                    npt.assert_equal(data.shape, (6, 2, 31))
                    npt.assert_almost_equal(data[:, 0, 0], np.array([0., 0., -2., 0.05, -0.05, 0.007]))

            else:
                npt.assert_almost_equal(f[0, 0], 7067.851604540217)

                # initial and final controls
                npt.assert_almost_equal(tau[:, 0], np.array([1.53525114e-09,  6.98419368e+00, -4.06669623e-10,  0.0]))
                npt.assert_almost_equal(tau[:, 10], np.array([-6.45031008e-10,  6.24337052e+00, -4.06669624e-10,  0.0]))
                npt.assert_almost_equal(tau[:, 20], np.array([-5.51259131e-10,  5.50254736e+00, -4.06669632e-10,  0.0]))
                npt.assert_almost_equal(tau[:, -1], np.array([1.64434894e-09,  4.83580652e+00, -4.06669638e-10,  0.0]))

                # initial and final position
                npt.assert_almost_equal(q[:, 0], np.array([-3.41506199e-15,  6.99773953e-16, -4.11684795e-12,  0.0]), decimal=5)
                npt.assert_almost_equal(q[:, -1], np.array([-3.84536375e-15,  2.00000000e+00,  6.27099006e-11,  0.0]), decimal=5)

                # how the force is stored
                data = ocp.nlp[0].numerical_data_timeseries["forces_in_global"]
                npt.assert_equal(data.shape, (9, 2, 31))
                npt.assert_almost_equal(data[:, 0, 0], np.array([0., 0., 0., 0., 0., -2., 0., 0., 0.]))

        else:
            if use_point_of_applications:
                if point_of_application_reference_frame == ReferenceFrame.LOCAL:
                    npt.assert_almost_equal(f[0, 0], 7076.043800572127)

                    # initial and final controls
                    npt.assert_almost_equal(tau[:, 0], np.array([0.01922272, 6.98428645, -0.28099094, 0.]))
                    npt.assert_almost_equal(tau[:, 10], np.array([4.40289150e-03, 6.24343465e+00, -2.16557082e-01, 0.0]))
                    npt.assert_almost_equal(tau[:, 20], np.array([-0.01041693, 5.50258285, -0.14650525, 0.]))
                    npt.assert_almost_equal(tau[:, -1], np.array([-0.02375477, 4.83581623, -0.12679319, 0.]))

                    # initial and final position
                    npt.assert_almost_equal(q[:, 0], np.array([-4.65561476e-15, 7.00215056e-16, -1.34683990e-03, 0.0]),
                                            decimal=5)
                    npt.assert_almost_equal(q[:, -1], np.array([-4.76138727e-15, 2.00000000e+00, 2.00652560e-03, 0.0]),
                                            decimal=5)

                    # how the force is stored
                    data = ocp.nlp[0].numerical_data_timeseries["forces_in_local"]
                    npt.assert_equal(data.shape, (9, 2, 31))
                    npt.assert_almost_equal(data[:, 0, 0], np.array([0., 0., 0., 0., 0., -2., 0.05, -0.05,
                                                                     0.007]))
            else:
                npt.assert_almost_equal(f[0, 0], 7067.851604540217)

                # initial and final controls
                npt.assert_almost_equal(tau[:, 0], np.array([1.92280974e-09,  6.98419368e+00, -4.00749051e-10,  0.0]))
                npt.assert_almost_equal(tau[:, 10], np.array([-8.87099972e-10,  6.24337052e+00,  2.22483147e-10,  0.0]))
                npt.assert_almost_equal(tau[:, 20], np.array([-6.85527926e-10,  5.50254736e+00,  1.70072550e-10,  0.0]))
                npt.assert_almost_equal(tau[:, -1], np.array([2.07070380e-09,  4.83580652e+00, -3.91202705e-10,  0.0]))

                # initial and final position
                npt.assert_almost_equal(q[:, 0], np.array([-4.64041726e-15,  6.99774094e-16,  4.75924448e-11,  0.0]),
                                        decimal=5)
                npt.assert_almost_equal(q[:, -1], np.array([-4.74298258e-15,  2.00000000e+00,  3.79354612e-11,  0.0]),
                                        decimal=5)

                # how the force is stored
                data = ocp.nlp[0].numerical_data_timeseries["forces_in_local"]
                npt.assert_equal(data.shape, (9, 2, 31))
                npt.assert_almost_equal(data[:, 0, 0], np.array([0., 0., 0., 0., 0., -2., 0.0, 0.0,
                                                                 0.0]))

    # detailed cost values
    npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], f[0, 0])



def prepare_ocp(
    biorbd_model_path: str = "models/cube_with_forces.bioMod",
    use_torque_and_force_at_the_same_time: bool = False,
) -> OptimalControlProgram:

    # Problem parameters
    n_shooting = 30
    final_time = 2

    # Linear external forces
    Seg1_force = np.zeros((3, n_shooting + 1))
    Seg1_force[2, :] = -2
    Seg1_force[2, 4] = -22

    Test_force = np.zeros((3, n_shooting + 1))
    Test_force[2, :] = 5
    Test_force[2, 4] = 52

    # Point of application
    Seg1_point_of_application = np.zeros((3, n_shooting + 1))
    Seg1_point_of_application[0, :] = 0.05
    Seg1_point_of_application[1, :] = -0.05
    Seg1_point_of_application[2, :] = 0.007

    Test_point_of_application = np.zeros((3, n_shooting + 1))
    Test_point_of_application[0, :] = -0.009
    Test_point_of_application[1, :] = 0.01
    Test_point_of_application[2, :] = -0.01

    # Torques external forces
    Seg1_torque = np.zeros((3, n_shooting + 1))
    Seg1_torque[1, :] = 1.8
    Seg1_torque[1, 4] = 18

    Test_torque = np.zeros((3, n_shooting + 1))
    Test_torque[1, :] = -1.8
    Test_torque[1, 4] = -18

    external_forces = ExternalForces()
    if use_torque_and_force_at_the_same_time:
        external_forces.add(
            key="Seg1",
            data=np.vstack((Seg1_torque, Seg1_force)),
            force_type=ExternalForceType.TORQUE_AND_FORCE,
            force_reference_frame=ReferenceFrame.GLOBAL,
            point_of_application=Seg1_point_of_application,
            point_of_application_reference_frame=ReferenceFrame.GLOBAL,
        )
        external_forces.add(
            key="Test",
            data=np.vstack((Test_torque, Test_force)),
            force_type=ExternalForceType.TORQUE_AND_FORCE,
            force_reference_frame=ReferenceFrame.LOCAL,
            point_of_application=Test_point_of_application,
            point_of_application_reference_frame=ReferenceFrame.LOCAL,
        )
    else:
        external_forces.add(
            key="Seg1",
            data=Seg1_force,
            force_type=ExternalForceType.FORCE,
            force_reference_frame=ReferenceFrame.GLOBAL,
            point_of_application=Seg1_point_of_application,
            point_of_application_reference_frame=ReferenceFrame.GLOBAL,
        )
        external_forces.add(
            key="Seg1",
            data=Seg1_torque,
            force_type=ExternalForceType.TORQUE,
            force_reference_frame=ReferenceFrame.GLOBAL,
            point_of_application=None,
            point_of_application_reference_frame=None,
        )
        external_forces.add(
            key="Test",
            data=Test_force,
            force_type=ExternalForceType.FORCE,
            force_reference_frame=ReferenceFrame.LOCAL,
            point_of_application=Test_point_of_application,
            point_of_application_reference_frame=ReferenceFrame.LOCAL,
        )
        external_forces.add(
            key="Test",
            data=Test_torque,
            force_type=ExternalForceType.TORQUE,
            force_reference_frame=ReferenceFrame.LOCAL,
            point_of_application=None,
            point_of_application_reference_frame=None,
        )

    bio_model = BiorbdModel(biorbd_model_path, external_forces=external_forces)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.TORQUE_DRIVEN,
        external_forces=external_forces,
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

def test_example_external_forces_all_at_once():

    from bioptim.examples.getting_started import example_external_forces as ocp_module
    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp_false = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_with_forces.bioMod",
        use_torque_and_force_at_the_same_time=False,
    )
    sol_false = ocp_false.solve()

    # Check objective function value
    f_false = np.array(sol_false.cost)
    npt.assert_equal(f_false.shape, (1, 1))
    npt.assert_almost_equal(f_false[0, 0], 7075.438561173094)

    # Check constraints
    g_false = np.array(sol_false.constraints)
    npt.assert_equal(g_false.shape, (246, 1))
    npt.assert_almost_equal(g_false, np.zeros((246, 1)))

    # Check some of the results
    states_false = sol_false.decision_states(to_merge=SolutionMerge.NODES)
    controls_false = sol_false.decision_controls(to_merge=SolutionMerge.NODES)
    q_false, qdot_false, tau_false = states_false["q"], states_false["qdot"], controls_false["tau"]

    # initial and final controls
    npt.assert_almost_equal(tau_false[:, 0], np.array([0.18595882,  6.98424521, -0.30071214,  0.]))
    npt.assert_almost_equal(tau_false[:, 10], np.array([-0.05224518,  6.24340374, -0.17010067,  0.]))
    npt.assert_almost_equal(tau_false[:, 20], np.array([-0.03359529,  5.50256227, -0.11790254,  0.]))
    npt.assert_almost_equal(tau_false[:, -1], np.array([0.07805599,  4.83580495, -0.14719148,  0.]))

    # initial and final position
    npt.assert_almost_equal(q_false[:, 0], np.array([-4.01261246e-15,  7.00046055e-16,  2.15375856e-03,  0.0]), decimal=5)
    npt.assert_almost_equal(q_false[:, -1], np.array([-4.29786206e-15,  2.00000000e+00,  1.66939403e-03,  0.0]), decimal=5)

    # initial and final velocities
    npt.assert_almost_equal(qdot_false[:, 0], np.array([0.0, 0.0, 0.0, 0.0]), decimal=5)
    npt.assert_almost_equal(qdot_false[:, -1], np.array([0.0, 0.0, 0.0, 0.0]), decimal=5)

    # how the force is stored
    data_Seg1_false = ocp_false.nlp[0].numerical_data_timeseries["forces_in_global"]
    npt.assert_equal(data_Seg1_false.shape, (9, 1, 31))
    npt.assert_almost_equal(data_Seg1_false[:, 0, 0], np.array([0.   ,  1.8  ,  0.   ,  0.   ,  0.   , -2.   ,  0.05 , -0.05 ,
        0.007]))

    data_Test_false = ocp_false.nlp[0].numerical_data_timeseries["forces_in_local"]
    npt.assert_equal(data_Test_false.shape, (9, 1, 31))
    npt.assert_almost_equal(data_Test_false[:, 0, 0], np.array([0. , -1.8,  0. ,  0. ,  0. ,  5. ,  -0.009,  0.01 ,
       -0.01]))

    # detailed cost values
    npt.assert_almost_equal(sol_false.detailed_cost[0]["cost_value_weighted"], f_false[0, 0])


    ocp_true = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_with_forces.bioMod",
        use_torque_and_force_at_the_same_time=True,
    )
    sol_true = ocp_true.solve()

    # Check objective function value
    f_true = np.array(sol_true.cost)
    npt.assert_equal(f_true.shape, (1, 1))
    npt.assert_almost_equal(f_true[0, 0], f_false[0, 0])

    # Check constraints
    g_true = np.array(sol_true.constraints)
    npt.assert_equal(g_true.shape, (246, 1))
    npt.assert_almost_equal(g_true, np.zeros((246, 1)))

    # Check some of the results
    states_true = sol_true.decision_states(to_merge=SolutionMerge.NODES)
    controls_true = sol_true.decision_controls(to_merge=SolutionMerge.NODES)
    q_true, qdot_true, tau_true = states_true["q"], states_true["qdot"], controls_true["tau"]

    # initial and final controls
    npt.assert_almost_equal(tau_true[:, 0], tau_false[:, 0])
    npt.assert_almost_equal(tau_true[:, 10], tau_false[:, 10])
    npt.assert_almost_equal(tau_true[:, 20], tau_false[:, 20])
    npt.assert_almost_equal(tau_true[:, -1], tau_false[:, -1])

    # initial and final position
    npt.assert_almost_equal(q_true[:, 0], q_true[:, 0])
    npt.assert_almost_equal(q_true[:, -1], q_true[:, -1])

    # initial and final velocities
    npt.assert_almost_equal(qdot_true[:, 0], qdot_true[:, 0])
    npt.assert_almost_equal(qdot_true[:, -1], qdot_true[:, -1])

    # how the force is stored
    data_Seg1_true = ocp_true.nlp[0].numerical_data_timeseries["forces_in_global"]
    npt.assert_equal(data_Seg1_true.shape, (9, 1, 31))
    npt.assert_almost_equal(data_Seg1_true[:, 0, 0], data_Seg1_false[:, 0, 0])

    data_Test_true = ocp_true.nlp[0].numerical_data_timeseries["forces_in_local"]
    npt.assert_equal(data_Test_true.shape, (9, 1, 31))
    npt.assert_almost_equal(data_Test_true[:, 0, 0], data_Test_false[:, 0, 0])

    # detailed cost values
    npt.assert_almost_equal(sol_true.detailed_cost[0]["cost_value_weighted"], f_true[0, 0])


def test_fail():

    n_shooting = 30
    fake_force = np.zeros((3, n_shooting + 1))

    external_forces = ExternalForces()
    external_forces.add(
        key="Seg1",
        data=np.vstack((fake_force, fake_force)),
        force_type=ExternalForceType.TORQUE_AND_FORCE,
        force_reference_frame=ReferenceFrame.GLOBAL,
    )
    external_forces.add(
        key="Test",
        data=np.vstack((fake_force, fake_force)),
        force_type=ExternalForceType.TORQUE_AND_FORCE,
        force_reference_frame=ReferenceFrame.LOCAL,
    )

    with pytest.raises(ValueError, match="The force is already defined for Seg1"):
        external_forces.add(
            key="Seg1",
            data=fake_force,
            force_type=ExternalForceType.FORCE,
            force_reference_frame=ReferenceFrame.GLOBAL,
        )

    with pytest.raises(ValueError, match="The torque is already defined for Seg1"):
        external_forces.add(
            key="Seg1",
            data=fake_force,
            force_type=ExternalForceType.TORQUE,
            force_reference_frame=ReferenceFrame.GLOBAL,
        )