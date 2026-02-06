import platform
import numpy as np
import numpy.testing as npt
import pytest
from casadi import DM, MX

from bioptim import (
    HolonomicBiorbdModel,
    HolonomicConstraintsFcn,
    HolonomicConstraintsList,
    Solver,
    SolutionMerge,
    OdeSolver,
)
from ..utils import TestUtils


def test_model_holonomic():

    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model_path = bioptim_folder + "/examples/models/triple_pendulum.bioMod"
    model = HolonomicBiorbdModel(biorbd_model_path)

    holonomic_constrains = HolonomicConstraintsList()
    holonomic_constrains.add(
        "y",
        HolonomicConstraintsFcn.superimpose_markers,
        marker_1="marker_1",
        marker_2="marker_6",
        index=slice(1, 2),
    )
    holonomic_constrains.add(
        "z",
        HolonomicConstraintsFcn.superimpose_markers,
        marker_1="marker_1",
        marker_2="marker_6",
        index=slice(2, 3),
    )

    with pytest.raises(
        ValueError,
        match="The sum of the number of dependent and independent joints "
        "should be equal to the number of DoF of the model",
    ):
        model.set_holonomic_configuration(holonomic_constrains, [1])

    with pytest.raises(
        ValueError,
        match="The sum of the number of dependent and independent joints should be equal to the number of DoF of the"
        " model",
    ):
        model.set_holonomic_configuration(holonomic_constrains, [1], [2])

    with pytest.raises(
        ValueError,
        match="Joint 1 is both dependant and independent. You need to specify this index in "
        "only one of these arguments: dependent_joint_index: independent_joint_index.",
    ):
        model.set_holonomic_configuration(holonomic_constrains, [1, 2], [1])

    with pytest.raises(
        ValueError,
        match="Joint index 3 is not a valid joint index since the model has 3 DoF",
    ):
        model.set_holonomic_configuration(holonomic_constrains, [1, 2], [3])

    with pytest.raises(
        ValueError,
        match="The dependent_joint_index should be sorted in ascending order.",
    ):
        model.set_holonomic_configuration(holonomic_constrains, [2, 1], [0])

    with pytest.raises(
        ValueError,
        match="The independent_joint_index should be sorted in ascending order.",
    ):
        model.set_holonomic_configuration(holonomic_constrains, [0], [2, 1])

    model.set_holonomic_configuration(holonomic_constrains, [1, 2], [0])

    with pytest.raises(
        ValueError,
        match="Length of state u size should be: 1. Got: 3",
    ):
        model.state_from_partition(MX([1, 2, 3]), MX([4]))

    with pytest.raises(
        ValueError,
        match="Length of state v size should be: 2. Got: 3",
    ):
        model.state_from_partition(MX([1]), MX([4, 5, 3]))

    npt.assert_equal(model.nb_independent_joints, 1)
    npt.assert_equal(model.nb_dependent_joints, 2)
    npt.assert_equal(model.nb_holonomic_constraints, 2)

    # symbolic variables
    q = MX([1, 2, 3])
    q_dot = MX([4, 5, 6])
    q_ddot = MX([7, 8, 9])
    tau = MX([10, 11, 12])

    q_u = MX(TestUtils.to_array(q[model._independent_joint_index]))
    qdot_u = MX(TestUtils.to_array(q_dot[model._independent_joint_index]))
    q_v = MX(TestUtils.to_array(q[model._dependent_joint_index]))
    q_ddot_u = MX(TestUtils.to_array(q_ddot[model._independent_joint_index]))

    # Test partition_coordinates
    output = model.partition_coordinates()
    TestUtils.assert_equal(output[0], [0])
    TestUtils.assert_equal(output[1], [1, 2])
    TestUtils.assert_equal(output[2], [1])

    # Test partitioned_forward_dynamics_with_qv
    TestUtils.assert_equal(
        model.partitioned_forward_dynamics_with_qv()(q_u, q_v[0], qdot_u, tau), [-5.180203], expand=False
    )

    # Test partitioned_forward_dynamics_full
    TestUtils.assert_equal(model.partitioned_forward_dynamics_full()(q, qdot_u, tau), [81.55801], expand=False)

    # Test error message for non-square Jacobian
    ill_model = HolonomicBiorbdModel(biorbd_model_path)
    ill_hconstraints = HolonomicConstraintsList()
    ill_hconstraints.add(
        "y",
        HolonomicConstraintsFcn.superimpose_markers,
        marker_1="marker_1",
        marker_2="marker_6",
        index=slice(1, 2),
    )
    with pytest.raises(
        ValueError,
        match=r"The shape of the dependent joint Jacobian should be square\. Got: \(1, 2\)\."
        r"Please consider checking the dimension of the holonomic constraints Jacobian\.\n"
        r"Here is a recommended partitioning: "
        r"      - independent_joint_index: \[1 2\],"
        r"      - dependent_joint_index: \[0\]\.",
    ):
        ill_model.set_holonomic_configuration(ill_hconstraints, [1, 2], [0])

    TestUtils.assert_equal(model.holonomic_constraints(q), [-0.70317549, 0.5104801])
    TestUtils.assert_equal(
        model.holonomic_constraints_jacobian(q),
        [[-0.5104801, 0.02982221, -0.96017029], [-0.70317549, 0.13829549, 0.2794155]],
    )
    TestUtils.assert_equal(model.holonomic_constraints_derivative(q, q_dot), [-7.65383105, -0.44473154])
    TestUtils.assert_equal(model.holonomic_constraints_double_derivative(q, q_dot, q_ddot), [-49.950546, -145.794884])
    TestUtils.assert_equal(
        model.constrained_forward_dynamics()(q, q_dot, tau, []), [-159.968193, 131.282108, 49.576071]
    )
    TestUtils.assert_equal(
        model.partitioned_mass_matrix(q),
        [
            [2.87597472e00, 4.60793003e-01, 3.36615631e-01],
            [4.60793003e-01, 9.99942366e-01, -2.88168107e-05],
            [3.36615631e-01, -2.88168107e-05, 9.54331080e-01],
        ],
    )
    TestUtils.assert_equal(model.partitioned_non_linear_effect(q, q_dot), [88.75493352, 4.13246046, -10.90514929])
    TestUtils.assert_equal(model.partitioned_q(q), [1.0, 2.0, 3.0])
    TestUtils.assert_equal(model.partitioned_qdot(q_dot), [4.0, 5.0, 6.0])
    TestUtils.assert_equal(model.partitioned_tau(tau), [10.0, 11.0, 12.0])
    TestUtils.assert_equal(
        model.partitioned_constraints_jacobian(q),
        [[-0.5104801, 0.02982221, -0.96017029], [-0.70317549, 0.13829549, 0.2794155]],
    )

    TestUtils.assert_equal(model.partitioned_forward_dynamics()(q_u, qdot_u, q_v, tau), -3.706354, expand=False)
    TestUtils.assert_equal(model.coupling_matrix(q), [5.79509793, -0.35166415], expand=False)
    TestUtils.assert_equal(model.biais_vector(q, q_dot), [1058.313451, -6.679008], expand=False)
    TestUtils.assert_equal(model.state_from_partition(q_u, q_v), q)

    TestUtils.assert_equal(model.compute_q_v()(q_u, q_v), [2 * np.pi / 3, 2 * np.pi / 3], expand=False)
    TestUtils.assert_equal(model.compute_q()(q_u, q_v), [1.0, 2.0943951, 2.0943951], expand=False)
    TestUtils.assert_equal(model.compute_qdot_v()(q, qdot_u), [23.18039172, -1.4066566], expand=False)
    TestUtils.assert_equal(model.compute_qdot()(q, qdot_u), [4.0, 23.18039172, -1.4066566], expand=False)

    qddot_v_expected = [1098.879137, -9.140657]
    TestUtils.assert_equal(model.compute_qddot_v()(q, q_dot, q_ddot_u), qddot_v_expected, expand=False)
    TestUtils.assert_equal(model.compute_qddot()(q, q_dot, q_ddot_u), [7.0] + qddot_v_expected, expand=False)

    npt.assert_almost_equal(
        model.compute_q_v()(DM([0.0]), DM([1.0, 1.0])).toarray().squeeze(),
        np.array([2 * np.pi / 3, 2 * np.pi / 3]),
        decimal=6,
    )

    TestUtils.assert_equal(
        model._compute_the_lagrangian_multipliers()(q, q_dot, q_ddot, tau), [20.34808, 27.119224], expand=False
    )
    TestUtils.assert_equal(
        model.compute_the_lagrangian_multipliers()(
            MX(np.zeros(model.nb_independent_joints)),
            MX(np.ones(model.nb_independent_joints) * 0.001),
            MX(np.zeros(model.nb_dependent_joints)),
            tau,
        ),
        [np.nan, np.nan],
        expand=False,
    )


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4(), OdeSolver.COLLOCATION()])
def test_example_two_pendulums(ode_solver):
    """Test the holonomic_constraints/two_pendulums example"""
    from bioptim.examples.toy_examples.holonomic_constraints import two_pendulums

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Prepare the ocp --- #
    ocp, model = two_pendulums.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/two_pendulums.bioMod",
        n_shooting=10,
        final_time=1,
        expand_dynamics=False,
        ode_solver=ode_solver,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)

    if isinstance(ode_solver, OdeSolver.RK4):
        npt.assert_almost_equal(
            states["q_u"],
            [
                [
                    1.54,
                    1.46024284,
                    1.24947784,
                    0.98555579,
                    0.69914724,
                    0.37122704,
                    -0.03002195,
                    -0.51108564,
                    -1.00708886,
                    -1.38595887,
                    -1.54,
                ],
                [
                    1.54,
                    1.63198614,
                    1.86553028,
                    2.11629728,
                    2.30191613,
                    2.37335671,
                    2.26406118,
                    1.91453347,
                    1.34329718,
                    0.66587232,
                    0.0,
                ],
            ],
            decimal=6,
        )

    elif isinstance(ode_solver, OdeSolver.COLLOCATION):
        npt.assert_almost_equal(
            states["q_u"],
            [
                [
                    1.54000000e00,
                    1.53960704e00,
                    1.53119417e00,
                    1.50384285e00,
                    1.47075073e00,
                    1.46024237e00,
                    1.44907691e00,
                    1.40171151e00,
                    1.32894631e00,
                    1.26675802e00,
                    1.24947642e00,
                    1.23197591e00,
                    1.16502476e00,
                    1.07516835e00,
                    1.00460860e00,
                    9.85554734e-01,
                    9.66412104e-01,
                    8.93831885e-01,
                    7.96802946e-01,
                    7.20030415e-01,
                    6.99146346e-01,
                    6.78066966e-01,
                    5.97057646e-01,
                    4.86014637e-01,
                    3.96011393e-01,
                    3.71226227e-01,
                    3.46078491e-01,
                    2.48270204e-01,
                    1.11985901e-01,
                    6.46711474e-04,
                    -3.00226953e-02,
                    -6.11063003e-02,
                    -1.81341284e-01,
                    -3.45693192e-01,
                    -4.75959406e-01,
                    -5.11086426e-01,
                    -5.46313411e-01,
                    -6.78694841e-01,
                    -8.49012892e-01,
                    -9.74688088e-01,
                    -1.00708982e00,
                    -1.03891405e00,
                    -1.15218687e00,
                    -1.28236238e00,
                    -1.36618443e00,
                    -1.38595972e00,
                    -1.40460611e00,
                    -1.46407898e00,
                    -1.51646647e00,
                    -1.53726816e00,
                    -1.54000000e00,
                ],
                [
                    1.54000000e00,
                    1.54045587e00,
                    1.55019337e00,
                    1.58181191e00,
                    1.61992431e00,
                    1.63198694e00,
                    1.64477139e00,
                    1.69858129e00,
                    1.77966388e00,
                    1.84713061e00,
                    1.86553257e00,
                    1.88397346e00,
                    1.95230605e00,
                    2.03811167e00,
                    2.10032450e00,
                    2.11629903e00,
                    2.13194694e00,
                    2.18720875e00,
                    2.25063538e00,
                    2.29202647e00,
                    2.30191745e00,
                    2.31126855e00,
                    2.34100320e00,
                    2.36597396e00,
                    2.37329336e00,
                    2.37335756e00,
                    2.37255179e00,
                    2.36109776e00,
                    2.32464067e00,
                    2.27895230e00,
                    2.26406163e00,
                    2.24798307e00,
                    2.17667837e00,
                    2.05757889e00,
                    1.94679445e00,
                    1.91453356e00,
                    1.88116543e00,
                    1.74655257e00,
                    1.55089225e00,
                    1.38817903e00,
                    1.34329686e00,
                    1.29790110e00,
                    1.12405118e00,
                    8.92207951e-01,
                    7.13412142e-01,
                    6.65871739e-01,
                    6.18449247e-01,
                    4.42108988e-01,
                    2.16115680e-01,
                    4.53319197e-02,
                    0.00000000e00,
                ],
            ],
            decimal=6,
        )


def test_example_two_pendulums_algebraic():
    """Test the holonomic_constraints/two_pendulums_algebraic example"""
    from bioptim.examples.toy_examples.holonomic_constraints import two_pendulums_algebraic

    if platform.system() == "Windows":
        pytest.skip("This test is skipped on Windows because too sensitive.")

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Prepare the ocp --- #
    ocp = two_pendulums_algebraic.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/two_pendulums.bioMod",
        n_shooting=5,
        final_time=1,
        expand_dynamics=False,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    algebraic_states = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)

    qu = states["q_u"]
    qdot_u = states["qdot_u"]
    qv = algebraic_states["q_v"]

    npt.assert_almost_equal(
        qu,
        np.array(
            [
                [
                    1.54,
                    1.55277912,
                    1.64601545,
                    1.70149005,
                    1.75074821,
                    1.7729181,
                    1.73988943,
                    1.66962305,
                    1.32169463,
                    1.13725974,
                    0.93489343,
                    0.33248456,
                    0.09385697,
                    -0.24597585,
                    -1.18928361,
                    -1.54,
                ],
                [
                    1.54,
                    1.51288542,
                    1.24586886,
                    1.07751375,
                    0.89766314,
                    0.36810254,
                    0.16028788,
                    -0.06376381,
                    -0.64591559,
                    -0.84802856,
                    -1.02114973,
                    -1.22319089,
                    -1.19797412,
                    -0.94296141,
                    -0.25131073,
                    0.0,
                ],
            ]
        ),
        decimal=6,
    )

    npt.assert_almost_equal(
        qv,
        np.array(
            [
                [
                    0.99952583,
                    0.99983769,
                    0.99717238,
                    0.99147172,
                    0.98385231,
                    0.97964284,
                    0.98573779,
                    0.99512061,
                    0.96913428,
                    0.90748577,
                    0.80453579,
                    0.32639252,
                    0.09371923,
                    -0.24350292,
                    -0.92810248,
                    -0.99952583,
                ],
                [
                    -0.03079146,
                    -0.01801623,
                    0.07514821,
                    0.13032198,
                    0.17898223,
                    0.20074836,
                    0.16828846,
                    0.09866593,
                    -0.24653348,
                    -0.42008283,
                    -0.59390417,
                    -0.94523432,
                    -0.99559867,
                    -0.96990016,
                    -0.37232485,
                    -0.03079146,
                ],
            ]
        ),
        decimal=6,
    )

    npt.assert_almost_equal(
        qdot_u,
        np.array(
            [
                [
                    0.0,
                    0.72949413,
                    1.96200651,
                    2.13477406,
                    1.50770278,
                    -0.86771318,
                    -1.97956706,
                    -3.37403326,
                    -6.66979418,
                    -7.68799241,
                    -8.17163582,
                    -9.21840884,
                    -9.50105648,
                    -13.4583828,
                    -13.77256442,
                    -10.04523499,
                ],
                [
                    0.0,
                    -1.81542041,
                    -5.89268312,
                    -7.06202618,
                    -7.239922,
                    -8.04717449,
                    -8.46022852,
                    -8.71929446,
                    -8.08597827,
                    -7.36329269,
                    -5.7789741,
                    -0.05345139,
                    2.55360354,
                    10.03655025,
                    9.92968345,
                    2.36850482,
                ],
            ]
        ),
        decimal=6,
    )


def test_example_three_bar():
    """Test the holonomic_constraints/three_bar example"""
    from bioptim.examples.toy_examples.holonomic_constraints import three_bar

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Prepare the ocp --- #
    ocp, model = three_bar.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/3bar.bioMod",
        n_shooting=10,
        final_time=1,
        expand_dynamics=False,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(
        states["q_u"],
        [
            [
                1.3,
                1.285379,
                1.241619,
                1.169068,
                1.068419,
                0.940879,
                0.78836,
                0.613674,
                0.420657,
                0.214183,
                0.0,
            ]
        ],
        decimal=6,
    )


def test_example_four_bar():
    """Test the holonomic_constraints/four_bar example"""
    from bioptim.examples.toy_examples.holonomic_constraints import four_bar

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Prepare the ocp --- #
    ocp, model = four_bar.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/4bar.bioMod",
        n_shooting=30,
        final_time=1,
        expand_dynamics=False,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(
        states["q_u"],
        [
            [
                0.77,
                0.76949359,
                0.76787468,
                0.76492894,
                0.7604129,
                0.75405472,
                0.74555372,
                0.7345782,
                0.72076059,
                0.70368875,
                0.68289166,
                0.65781596,
                0.627787,
                0.59193902,
                0.54907238,
                0.49730222,
                0.43305918,
                0.34900314,
                0.24145408,
                0.12931534,
                0.02311091,
                -0.07736037,
                -0.17257584,
                -0.26263321,
                -0.34766303,
                -0.42792951,
                -0.50378263,
                -0.57559366,
                -0.64371071,
                -0.70843351,
                -0.77,
            ],
            [
                0.0,
                0.00033752,
                0.00123364,
                0.00244202,
                0.0036886,
                0.0046699,
                0.0050507,
                0.00446145,
                0.00249508,
                -0.00129614,
                -0.00740281,
                -0.01635682,
                -0.02871775,
                -0.04502648,
                -0.06566813,
                -0.09048189,
                -0.11763193,
                -0.14049351,
                -0.14563252,
                -0.13032206,
                -0.10420413,
                -0.07458985,
                -0.04603899,
                -0.02128835,
                -0.00176531,
                0.01196416,
                0.01985102,
                0.02214153,
                0.01924878,
                0.01168099,
                0.0,
            ],
        ],
        decimal=6,
    )


def test_example_two_pendulums_2constraint_4DOF():
    """Test the holonomic_constraints/two_pendulums example"""
    from bioptim.examples.toy_examples.holonomic_constraints import two_pendulums_2constraint_4DOF

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Prepare the ocp --- #
    ocp, model = two_pendulums_2constraint_4DOF.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/two_pendulums.bioMod",
        n_shooting=10,
        final_time=1,
        expand_dynamics=False,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(
        states["q_u"],
        [
            [
                -0.5,
                -0.47400031,
                -0.40886377,
                -0.31068409,
                -0.18888455,
                -0.05543956,
                0.07635419,
                0.19339095,
                0.28422433,
                0.34019268,
                0.35608811,
            ]
        ],
        decimal=6,
    )


def test_example_two_pendulums_2constraint():
    """Test the holonomic_constraints/two_pendulums example"""
    from bioptim.examples.toy_examples.holonomic_constraints import two_pendulums_2constraint

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Prepare the ocp --- #
    ocp, model = two_pendulums_2constraint.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/two_pendulums_2.bioMod",
        n_shooting=10,
        final_time=1,
        expand_dynamics=False,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(
        states["q_u"],
        [
            [
                -0.5,
                -0.476922,
                -0.409191,
                -0.301994,
                -0.164726,
                -0.010378,
                0.145734,
                0.28777,
                0.401256,
                0.474597,
                0.5,
            ]
        ],
        decimal=6,
    )


def test_example_two_pendulums_rotule():
    """Test the holonomic_constraints/two_pendulums example"""
    from bioptim.examples.toy_examples.holonomic_constraints import two_pendulums_rotule

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Prepare the ocp --- #
    ocp, model = two_pendulums_rotule.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/two_pendulums_rotule.bioMod",
        n_shooting=10,
        final_time=1,
        expand_dynamics=False,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(
        states["q_u"],
        [
            [
                0.523599,
                0.497571,
                0.41848,
                0.286916,
                0.110176,
                -0.09611,
                -0.30917,
                -0.503139,
                -0.654991,
                -0.750469,
                -0.785398,
            ]
        ],
        decimal=6,
    )


def test_example_arm26_pendulum_swingup():
    """Test the holonomic_constraints/two_pendulums example"""
    from bioptim.examples.toy_examples.holonomic_constraints import arm26_pendulum_swingup

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Prepare the ocp --- #
    ocp, model = arm26_pendulum_swingup.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/arm26_w_pendulum.bioMod",
        n_shooting=30,
        final_time=1,
        expand_dynamics=False,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(
        states["q_u"],
        [
            [
                -2.69135824e-01,
                -2.64498120e-01,
                -2.50200880e-01,
                -2.25632471e-01,
                -1.90588426e-01,
                -1.45509776e-01,
                -9.17706792e-02,
                -3.18630768e-02,
                2.15488905e-02,
                3.71700043e-02,
                -1.47743514e-02,
                -1.18632803e-01,
                -2.25113739e-01,
                -3.08059286e-01,
                -3.62876634e-01,
                -3.92234880e-01,
                -4.01385872e-01,
                -3.95867834e-01,
                -3.79002763e-01,
                -3.50727200e-01,
                -3.08796279e-01,
                -2.50925016e-01,
                -1.76355800e-01,
                -8.66290583e-02,
                1.41840915e-02,
                1.19698408e-01,
                2.21938379e-01,
                3.12495334e-01,
                3.83726881e-01,
                4.29493011e-01,
                4.45350909e-01,
            ],
            [
                1.32697117e-04,
                3.85224467e-03,
                1.29670231e-02,
                2.31446707e-02,
                2.98803773e-02,
                2.93092213e-02,
                1.92421639e-02,
                4.25995214e-08,
                1.98533259e-07,
                9.97417517e-02,
                3.78068308e-01,
                7.97638080e-01,
                1.23671744e00,
                1.63341338e00,
                1.97730469e00,
                2.27031542e00,
                2.51487477e00,
                2.71245371e00,
                2.86470161e00,
                2.97447848e00,
                3.04579429e00,
                3.08319634e00,
                3.09131402e00,
                3.07465082e00,
                3.03752423e00,
                2.98423604e00,
                2.91965360e00,
                2.85018330e00,
                2.78474313e00,
                2.73519685e00,
                2.71606423e00,
            ],
            [
                0.00000000e00,
                9.29978638e-03,
                3.61965675e-02,
                7.78802301e-02,
                1.29945901e-01,
                1.86553495e-01,
                2.40873672e-01,
                2.85784980e-01,
                3.15907293e-01,
                3.28332167e-01,
                3.10026789e-01,
                2.28632607e-01,
                7.48871946e-02,
                -1.30511638e-01,
                -3.68045837e-01,
                -6.25640028e-01,
                -8.94498746e-01,
                -1.16550296e00,
                -1.42894095e00,
                -1.67707730e00,
                -1.90606183e00,
                -2.11529404e00,
                -2.30561703e00,
                -2.47795985e00,
                -2.63280431e00,
                -2.77016185e00,
                -2.88962790e00,
                -2.99014292e00,
                -3.06928988e00,
                -3.12234802e00,
                -3.14159265e00,
            ],
        ],
        decimal=6,
    )


def test_example_arm26_pendulum_swingup_muscle():
    """Test the holonomic_constraints/two_pendulums example"""
    from bioptim.examples.toy_examples.holonomic_constraints import arm26_pendulum_swingup_muscle

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Prepare the ocp --- #
    ocp, model = arm26_pendulum_swingup_muscle.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/arm26_w_pendulum.bioMod",
        n_shooting=30,
        final_time=1,
        expand_dynamics=False,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(
        states["q_u"],
        [
            [
                0.00000000e00,
                -3.20620080e-05,
                -7.22737278e-04,
                -2.95404501e-03,
                -5.63924692e-03,
                -6.49005147e-03,
                -7.44120424e-03,
                -1.23041824e-02,
                -2.16864947e-02,
                -3.11639947e-02,
                -3.40183367e-02,
                -3.70432103e-02,
                -5.02028401e-02,
                -7.15901095e-02,
                -9.11251000e-02,
                -9.67782100e-02,
                -1.02636806e-01,
                -1.26605153e-01,
                -1.62430055e-01,
                -1.93215657e-01,
                -2.01885869e-01,
                -2.10736867e-01,
                -2.45435943e-01,
                -2.94093000e-01,
                -3.33840955e-01,
                -3.44774564e-01,
                -3.55823516e-01,
                -3.98072073e-01,
                -4.54910190e-01,
                -4.99606632e-01,
                -5.11656422e-01,
                -5.23739586e-01,
                -5.69147128e-01,
                -6.28166932e-01,
                -6.72816216e-01,
                -6.84573949e-01,
                -6.96250019e-01,
                -7.39135696e-01,
                -7.92336556e-01,
                -8.30465979e-01,
                -8.40175985e-01,
                -8.49677979e-01,
                -8.83314439e-01,
                -9.21855611e-01,
                -9.46785879e-01,
                -9.52697267e-01,
                -9.58289149e-01,
                -9.76331668e-01,
                -9.92490110e-01,
                -9.98834344e-01,
                -9.99597271e-01,
                -9.99995036e-01,
                -9.98551302e-01,
                -9.89799356e-01,
                -9.77968499e-01,
                -9.74086523e-01,
                -9.69981207e-01,
                -9.53456348e-01,
                -9.29990775e-01,
                -9.11227010e-01,
                -9.06184652e-01,
                -9.01122130e-01,
                -8.81898582e-01,
                -8.56759436e-01,
                -8.37838636e-01,
                -8.32883913e-01,
                -8.27946707e-01,
                -8.09404808e-01,
                -7.85340850e-01,
                -7.67085834e-01,
                -7.62252280e-01,
                -7.57419485e-01,
                -7.39182477e-01,
                -7.15118888e-01,
                -6.96416053e-01,
                -6.91386302e-01,
                -6.86330724e-01,
                -6.67071745e-01,
                -6.41203964e-01,
                -6.20752421e-01,
                -6.15205311e-01,
                -6.09614819e-01,
                -5.88228319e-01,
                -5.59333655e-01,
                -5.36409182e-01,
                -5.30185937e-01,
                -5.23914212e-01,
                -4.99944540e-01,
                -4.67660510e-01,
                -4.42166181e-01,
                -4.35267039e-01,
                -4.28324533e-01,
                -4.01895309e-01,
                -3.66576993e-01,
                -3.38924873e-01,
                -3.31478891e-01,
                -3.24002617e-01,
                -2.95697827e-01,
                -2.58261881e-01,
                -2.29261624e-01,
                -2.21498827e-01,
                -2.13725121e-01,
                -1.84491458e-01,
                -1.46302787e-01,
                -1.17083535e-01,
                -1.09314554e-01,
                -1.01558643e-01,
                -7.26247823e-02,
                -3.53760500e-02,
                -7.29152103e-03,
                1.15378024e-04,
                7.48066662e-03,
                3.46628706e-02,
                6.89600515e-02,
                9.42902119e-02,
                1.00893971e-01,
                1.07423714e-01,
                1.31152927e-01,
                1.60235905e-01,
                1.81077138e-01,
                1.86419341e-01,
                1.91658860e-01,
                2.10283440e-01,
                2.32176279e-01,
                2.47191245e-01,
                2.50945089e-01,
                2.54592465e-01,
                2.67307303e-01,
                2.81730339e-01,
                2.91261524e-01,
                2.93592960e-01,
                2.95870417e-01,
                3.04207591e-01,
                3.14515826e-01,
                3.21882445e-01,
                3.23752774e-01,
                3.25632869e-01,
                3.33139282e-01,
                3.43495577e-01,
                3.51382102e-01,
                3.53423056e-01,
                3.55450956e-01,
                3.63029886e-01,
                3.72308078e-01,
                3.78486893e-01,
                3.79950859e-01,
                3.81334667e-01,
                3.85790931e-01,
                3.89592744e-01,
                3.90787335e-01,
                3.90843002e-01,
            ],
            [
                1.57079633e00,
                1.57070683e00,
                1.56877430e00,
                1.56244859e00,
                1.55465633e00,
                1.55214408e00,
                1.54963079e00,
                1.54182624e00,
                1.53548604e00,
                1.53352726e00,
                1.53342243e00,
                1.53358406e00,
                1.53736159e00,
                1.54960770e00,
                1.56434639e00,
                1.56902273e00,
                1.57404700e00,
                1.59627475e00,
                1.63279146e00,
                1.66601729e00,
                1.67557322e00,
                1.68532662e00,
                1.72284498e00,
                1.77351446e00,
                1.81321380e00,
                1.82386568e00,
                1.83439811e00,
                1.87157611e00,
                1.91428524e00,
                1.94235745e00,
                1.94912944e00,
                1.95552450e00,
                1.97548072e00,
                1.99181022e00,
                1.99678133e00,
                1.99698930e00,
                1.99672126e00,
                1.99142238e00,
                1.97425977e00,
                1.95318876e00,
                1.94640747e00,
                1.93912560e00,
                1.90723862e00,
                1.85469384e00,
                1.80586461e00,
                1.79157553e00,
                1.77675767e00,
                1.71650563e00,
                1.62652338e00,
                1.54835936e00,
                1.52611510e00,
                1.50340802e00,
                1.41535923e00,
                1.29372722e00,
                1.19538561e00,
                1.16845027e00,
                1.14150204e00,
                1.04306198e00,
                9.21943120e-01,
                8.35634220e-01,
                8.13697318e-01,
                7.92169294e-01,
                7.14666065e-01,
                6.22231124e-01,
                5.58760093e-01,
                5.42994035e-01,
                5.27638109e-01,
                4.73047855e-01,
                4.09347337e-01,
                3.66429173e-01,
                3.55855200e-01,
                3.45601623e-01,
                3.09610923e-01,
                2.68406366e-01,
                2.41017169e-01,
                2.34301495e-01,
                2.27807912e-01,
                2.05218281e-01,
                1.79715374e-01,
                1.62960615e-01,
                1.58876766e-01,
                1.54942802e-01,
                1.41427677e-01,
                1.26563232e-01,
                1.17118160e-01,
                1.14868968e-01,
                1.12728071e-01,
                1.05624097e-01,
                9.84557299e-02,
                9.44808344e-02,
                9.36358519e-02,
                9.28804178e-02,
                9.08415615e-02,
                9.00335035e-02,
                9.08065201e-02,
                9.12144976e-02,
                9.17104157e-02,
                9.43848689e-02,
                9.98127571e-02,
                1.05468918e-01,
                1.07196552e-01,
                1.09024400e-01,
                1.16850090e-01,
                1.29379581e-01,
                1.40759110e-01,
                1.44048120e-01,
                1.47457779e-01,
                1.61451462e-01,
                1.82532621e-01,
                2.00814099e-01,
                2.05989285e-01,
                2.11314763e-01,
                2.32853359e-01,
                2.64595386e-01,
                2.91635511e-01,
                2.99224638e-01,
                3.07008184e-01,
                3.38258820e-01,
                3.83750693e-01,
                4.22063757e-01,
                4.32751871e-01,
                4.43680627e-01,
                4.87198283e-01,
                5.49624470e-01,
                6.01465572e-01,
                6.15820485e-01,
                6.30412117e-01,
                6.87375487e-01,
                7.66460244e-01,
                8.30283417e-01,
                8.47710340e-01,
                8.65207631e-01,
                9.30593835e-01,
                1.01512930e00,
                1.07926290e00,
                1.09625261e00,
                1.11301976e00,
                1.17246183e00,
                1.24233209e00,
                1.29052647e00,
                1.30263879e00,
                1.31436154e00,
                1.35418626e00,
                1.39682255e00,
                1.42284132e00,
                1.42884489e00,
                1.43444363e00,
                1.45171211e00,
                1.46561777e00,
                1.46984373e00,
                1.47003907e00,
            ],
            [
                0.00000000e00,
                -3.69853872e-05,
                -8.38371941e-04,
                -3.46533231e-03,
                -6.66746507e-03,
                -7.68849847e-03,
                -8.83255584e-03,
                -1.46985317e-02,
                -2.60077458e-02,
                -3.73719424e-02,
                -4.07793805e-02,
                -4.43777462e-02,
                -5.98551189e-02,
                -8.44521151e-02,
                -1.06360344e-01,
                -1.12604114e-01,
                -1.19028804e-01,
                -1.44831438e-01,
                -1.82084258e-01,
                -2.12994678e-01,
                -2.21534824e-01,
                -2.30194881e-01,
                -2.63716563e-01,
                -3.09709927e-01,
                -3.46544679e-01,
                -3.56576492e-01,
                -3.66702404e-01,
                -4.05603813e-01,
                -4.58428783e-01,
                -5.00379502e-01,
                -5.11750893e-01,
                -5.23208283e-01,
                -5.66999155e-01,
                -6.25948082e-01,
                -6.72529109e-01,
                -6.85155148e-01,
                -6.97865587e-01,
                -7.46286656e-01,
                -8.11049720e-01,
                -8.61690577e-01,
                -8.75295808e-01,
                -8.88942417e-01,
                -9.40485507e-01,
                -1.00830794e00,
                -1.06071058e00,
                -1.07475359e00,
                -1.08883098e00,
                -1.14196938e00,
                -1.21265328e00,
                -1.26859526e00,
                -1.28385569e00,
                -1.29919882e00,
                -1.35686588e00,
                -1.43254040e00,
                -1.49101266e00,
                -1.50666700e00,
                -1.52224141e00,
                -1.57892342e00,
                -1.64847044e00,
                -1.69837862e00,
                -1.71119877e00,
                -1.72384419e00,
                -1.76992481e00,
                -1.82724101e00,
                -1.86944804e00,
                -1.88048734e00,
                -1.89144961e00,
                -1.93200404e00,
                -1.98362784e00,
                -2.02229775e00,
                -2.03247896e00,
                -2.04261083e00,
                -2.08020291e00,
                -2.12833957e00,
                -2.16465355e00,
                -2.17425823e00,
                -2.18383389e00,
                -2.21952096e00,
                -2.26555730e00,
                -2.30048373e00,
                -2.30973997e00,
                -2.31897454e00,
                -2.35343101e00,
                -2.39788741e00,
                -2.43154523e00,
                -2.44044773e00,
                -2.44932129e00,
                -2.48235304e00,
                -2.52475640e00,
                -2.55667488e00,
                -2.56508894e00,
                -2.57346403e00,
                -2.60454108e00,
                -2.64420153e00,
                -2.67387865e00,
                -2.68167631e00,
                -2.68942782e00,
                -2.71810862e00,
                -2.75451948e00,
                -2.78161920e00,
                -2.78871806e00,
                -2.79576644e00,
                -2.82177689e00,
                -2.85463030e00,
                -2.87893907e00,
                -2.88528286e00,
                -2.89157116e00,
                -2.91468371e00,
                -2.94362591e00,
                -2.96481876e00,
                -2.97031336e00,
                -2.97574440e00,
                -2.99556735e00,
                -3.02003833e00,
                -3.03766100e00,
                -3.04218284e00,
                -3.04663166e00,
                -3.06267949e00,
                -3.08200669e00,
                -3.09551393e00,
                -3.09891340e00,
                -3.10222757e00,
                -3.11388962e00,
                -3.12716503e00,
                -3.13576099e00,
                -3.13781028e00,
                -3.13975605e00,
                -3.14610970e00,
                -3.15207499e00,
                -3.15481161e00,
                -3.15526781e00,
                -3.15562439e00,
                -3.15622576e00,
                -3.15530450e00,
                -3.15332736e00,
                -3.15261784e00,
                -3.15188007e00,
                -3.14930693e00,
                -3.14646739e00,
                -3.14463929e00,
                -3.14418617e00,
                -3.14377182e00,
                -3.14277291e00,
                -3.14226273e00,
                -3.14204669e00,
                -3.14197530e00,
                -3.14190633e00,
                -3.14171569e00,
                -3.14160546e00,
                -3.14159152e00,
                -3.14159265e00,
            ],
        ],
        decimal=6,
    )


def test_example_arm26_pendulum_swingup_muscle_algebraic():
    """Test the holonomic_constraints/two_pendulums example"""
    from bioptim.examples.toy_examples.holonomic_constraints import arm26_pendulum_swingup_muscle_algebraic

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Prepare the ocp --- #
    ocp, model = arm26_pendulum_swingup_muscle_algebraic.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/arm26_w_pendulum.bioMod",
        n_shooting=30,
        final_time=1,
        expand_dynamics=False,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(
        states["q_u"],
        [
            [
                0.00000000e00,
                -3.20620080e-05,
                -7.22737278e-04,
                -2.95404501e-03,
                -5.63924692e-03,
                -6.49005147e-03,
                -7.44120424e-03,
                -1.23041824e-02,
                -2.16864947e-02,
                -3.11639947e-02,
                -3.40183367e-02,
                -3.70432103e-02,
                -5.02028401e-02,
                -7.15901095e-02,
                -9.11251000e-02,
                -9.67782100e-02,
                -1.02636806e-01,
                -1.26605153e-01,
                -1.62430055e-01,
                -1.93215657e-01,
                -2.01885869e-01,
                -2.10736867e-01,
                -2.45435943e-01,
                -2.94093000e-01,
                -3.33840955e-01,
                -3.44774564e-01,
                -3.55823516e-01,
                -3.98072073e-01,
                -4.54910190e-01,
                -4.99606632e-01,
                -5.11656422e-01,
                -5.23739586e-01,
                -5.69147128e-01,
                -6.28166932e-01,
                -6.72816216e-01,
                -6.84573949e-01,
                -6.96250019e-01,
                -7.39135696e-01,
                -7.92336556e-01,
                -8.30465979e-01,
                -8.40175985e-01,
                -8.49677979e-01,
                -8.83314439e-01,
                -9.21855611e-01,
                -9.46785879e-01,
                -9.52697267e-01,
                -9.58289149e-01,
                -9.76331668e-01,
                -9.92490110e-01,
                -9.98834344e-01,
                -9.99597271e-01,
                -9.99995036e-01,
                -9.98551302e-01,
                -9.89799356e-01,
                -9.77968499e-01,
                -9.74086523e-01,
                -9.69981207e-01,
                -9.53456348e-01,
                -9.29990775e-01,
                -9.11227010e-01,
                -9.06184652e-01,
                -9.01122130e-01,
                -8.81898582e-01,
                -8.56759436e-01,
                -8.37838636e-01,
                -8.32883913e-01,
                -8.27946707e-01,
                -8.09404808e-01,
                -7.85340850e-01,
                -7.67085834e-01,
                -7.62252280e-01,
                -7.57419485e-01,
                -7.39182477e-01,
                -7.15118888e-01,
                -6.96416053e-01,
                -6.91386302e-01,
                -6.86330724e-01,
                -6.67071745e-01,
                -6.41203964e-01,
                -6.20752421e-01,
                -6.15205311e-01,
                -6.09614819e-01,
                -5.88228319e-01,
                -5.59333655e-01,
                -5.36409182e-01,
                -5.30185937e-01,
                -5.23914212e-01,
                -4.99944540e-01,
                -4.67660510e-01,
                -4.42166181e-01,
                -4.35267039e-01,
                -4.28324533e-01,
                -4.01895309e-01,
                -3.66576993e-01,
                -3.38924873e-01,
                -3.31478891e-01,
                -3.24002617e-01,
                -2.95697827e-01,
                -2.58261881e-01,
                -2.29261624e-01,
                -2.21498827e-01,
                -2.13725121e-01,
                -1.84491458e-01,
                -1.46302787e-01,
                -1.17083535e-01,
                -1.09314554e-01,
                -1.01558643e-01,
                -7.26247823e-02,
                -3.53760500e-02,
                -7.29152103e-03,
                1.15378024e-04,
                7.48066662e-03,
                3.46628706e-02,
                6.89600515e-02,
                9.42902119e-02,
                1.00893971e-01,
                1.07423714e-01,
                1.31152927e-01,
                1.60235905e-01,
                1.81077138e-01,
                1.86419341e-01,
                1.91658860e-01,
                2.10283440e-01,
                2.32176279e-01,
                2.47191245e-01,
                2.50945089e-01,
                2.54592465e-01,
                2.67307303e-01,
                2.81730339e-01,
                2.91261524e-01,
                2.93592960e-01,
                2.95870417e-01,
                3.04207591e-01,
                3.14515826e-01,
                3.21882445e-01,
                3.23752774e-01,
                3.25632869e-01,
                3.33139282e-01,
                3.43495577e-01,
                3.51382102e-01,
                3.53423056e-01,
                3.55450956e-01,
                3.63029886e-01,
                3.72308078e-01,
                3.78486893e-01,
                3.79950859e-01,
                3.81334667e-01,
                3.85790931e-01,
                3.89592744e-01,
                3.90787335e-01,
                3.90843002e-01,
            ],
            [
                1.57079633e00,
                1.57070683e00,
                1.56877430e00,
                1.56244859e00,
                1.55465633e00,
                1.55214408e00,
                1.54963079e00,
                1.54182624e00,
                1.53548604e00,
                1.53352726e00,
                1.53342243e00,
                1.53358406e00,
                1.53736159e00,
                1.54960770e00,
                1.56434639e00,
                1.56902273e00,
                1.57404700e00,
                1.59627475e00,
                1.63279146e00,
                1.66601729e00,
                1.67557322e00,
                1.68532662e00,
                1.72284498e00,
                1.77351446e00,
                1.81321380e00,
                1.82386568e00,
                1.83439811e00,
                1.87157611e00,
                1.91428524e00,
                1.94235745e00,
                1.94912944e00,
                1.95552450e00,
                1.97548072e00,
                1.99181022e00,
                1.99678133e00,
                1.99698930e00,
                1.99672126e00,
                1.99142238e00,
                1.97425977e00,
                1.95318876e00,
                1.94640747e00,
                1.93912560e00,
                1.90723862e00,
                1.85469384e00,
                1.80586461e00,
                1.79157553e00,
                1.77675767e00,
                1.71650563e00,
                1.62652338e00,
                1.54835936e00,
                1.52611510e00,
                1.50340802e00,
                1.41535923e00,
                1.29372722e00,
                1.19538561e00,
                1.16845027e00,
                1.14150204e00,
                1.04306198e00,
                9.21943120e-01,
                8.35634220e-01,
                8.13697318e-01,
                7.92169294e-01,
                7.14666065e-01,
                6.22231124e-01,
                5.58760093e-01,
                5.42994035e-01,
                5.27638109e-01,
                4.73047855e-01,
                4.09347337e-01,
                3.66429173e-01,
                3.55855200e-01,
                3.45601623e-01,
                3.09610923e-01,
                2.68406366e-01,
                2.41017169e-01,
                2.34301495e-01,
                2.27807912e-01,
                2.05218281e-01,
                1.79715374e-01,
                1.62960615e-01,
                1.58876766e-01,
                1.54942802e-01,
                1.41427677e-01,
                1.26563232e-01,
                1.17118160e-01,
                1.14868968e-01,
                1.12728071e-01,
                1.05624097e-01,
                9.84557299e-02,
                9.44808344e-02,
                9.36358519e-02,
                9.28804178e-02,
                9.08415615e-02,
                9.00335035e-02,
                9.08065201e-02,
                9.12144976e-02,
                9.17104157e-02,
                9.43848689e-02,
                9.98127571e-02,
                1.05468918e-01,
                1.07196552e-01,
                1.09024400e-01,
                1.16850090e-01,
                1.29379581e-01,
                1.40759110e-01,
                1.44048120e-01,
                1.47457779e-01,
                1.61451462e-01,
                1.82532621e-01,
                2.00814099e-01,
                2.05989285e-01,
                2.11314763e-01,
                2.32853359e-01,
                2.64595386e-01,
                2.91635511e-01,
                2.99224638e-01,
                3.07008184e-01,
                3.38258820e-01,
                3.83750693e-01,
                4.22063757e-01,
                4.32751871e-01,
                4.43680627e-01,
                4.87198283e-01,
                5.49624470e-01,
                6.01465572e-01,
                6.15820485e-01,
                6.30412117e-01,
                6.87375487e-01,
                7.66460244e-01,
                8.30283417e-01,
                8.47710340e-01,
                8.65207631e-01,
                9.30593835e-01,
                1.01512930e00,
                1.07926290e00,
                1.09625261e00,
                1.11301976e00,
                1.17246183e00,
                1.24233209e00,
                1.29052647e00,
                1.30263879e00,
                1.31436154e00,
                1.35418626e00,
                1.39682255e00,
                1.42284132e00,
                1.42884489e00,
                1.43444363e00,
                1.45171211e00,
                1.46561777e00,
                1.46984373e00,
                1.47003907e00,
            ],
            [
                0.00000000e00,
                -3.69853872e-05,
                -8.38371941e-04,
                -3.46533231e-03,
                -6.66746507e-03,
                -7.68849847e-03,
                -8.83255584e-03,
                -1.46985317e-02,
                -2.60077458e-02,
                -3.73719424e-02,
                -4.07793805e-02,
                -4.43777462e-02,
                -5.98551189e-02,
                -8.44521151e-02,
                -1.06360344e-01,
                -1.12604114e-01,
                -1.19028804e-01,
                -1.44831438e-01,
                -1.82084258e-01,
                -2.12994678e-01,
                -2.21534824e-01,
                -2.30194881e-01,
                -2.63716563e-01,
                -3.09709927e-01,
                -3.46544679e-01,
                -3.56576492e-01,
                -3.66702404e-01,
                -4.05603813e-01,
                -4.58428783e-01,
                -5.00379502e-01,
                -5.11750893e-01,
                -5.23208283e-01,
                -5.66999155e-01,
                -6.25948082e-01,
                -6.72529109e-01,
                -6.85155148e-01,
                -6.97865587e-01,
                -7.46286656e-01,
                -8.11049720e-01,
                -8.61690577e-01,
                -8.75295808e-01,
                -8.88942417e-01,
                -9.40485507e-01,
                -1.00830794e00,
                -1.06071058e00,
                -1.07475359e00,
                -1.08883098e00,
                -1.14196938e00,
                -1.21265328e00,
                -1.26859526e00,
                -1.28385569e00,
                -1.29919882e00,
                -1.35686588e00,
                -1.43254040e00,
                -1.49101266e00,
                -1.50666700e00,
                -1.52224141e00,
                -1.57892342e00,
                -1.64847044e00,
                -1.69837862e00,
                -1.71119877e00,
                -1.72384419e00,
                -1.76992481e00,
                -1.82724101e00,
                -1.86944804e00,
                -1.88048734e00,
                -1.89144961e00,
                -1.93200404e00,
                -1.98362784e00,
                -2.02229775e00,
                -2.03247896e00,
                -2.04261083e00,
                -2.08020291e00,
                -2.12833957e00,
                -2.16465355e00,
                -2.17425823e00,
                -2.18383389e00,
                -2.21952096e00,
                -2.26555730e00,
                -2.30048373e00,
                -2.30973997e00,
                -2.31897454e00,
                -2.35343101e00,
                -2.39788741e00,
                -2.43154523e00,
                -2.44044773e00,
                -2.44932129e00,
                -2.48235304e00,
                -2.52475640e00,
                -2.55667488e00,
                -2.56508894e00,
                -2.57346403e00,
                -2.60454108e00,
                -2.64420153e00,
                -2.67387865e00,
                -2.68167631e00,
                -2.68942782e00,
                -2.71810862e00,
                -2.75451948e00,
                -2.78161920e00,
                -2.78871806e00,
                -2.79576644e00,
                -2.82177689e00,
                -2.85463030e00,
                -2.87893907e00,
                -2.88528286e00,
                -2.89157116e00,
                -2.91468371e00,
                -2.94362591e00,
                -2.96481876e00,
                -2.97031336e00,
                -2.97574440e00,
                -2.99556735e00,
                -3.02003833e00,
                -3.03766100e00,
                -3.04218284e00,
                -3.04663166e00,
                -3.06267949e00,
                -3.08200669e00,
                -3.09551393e00,
                -3.09891340e00,
                -3.10222757e00,
                -3.11388962e00,
                -3.12716503e00,
                -3.13576099e00,
                -3.13781028e00,
                -3.13975605e00,
                -3.14610970e00,
                -3.15207499e00,
                -3.15481161e00,
                -3.15526781e00,
                -3.15562439e00,
                -3.15622576e00,
                -3.15530450e00,
                -3.15332736e00,
                -3.15261784e00,
                -3.15188007e00,
                -3.14930693e00,
                -3.14646739e00,
                -3.14463929e00,
                -3.14418617e00,
                -3.14377182e00,
                -3.14277291e00,
                -3.14226273e00,
                -3.14204669e00,
                -3.14197530e00,
                -3.14190633e00,
                -3.14171569e00,
                -3.14160546e00,
                -3.14159152e00,
                -3.14159265e00,
            ],
        ],
        decimal=6,
    )
