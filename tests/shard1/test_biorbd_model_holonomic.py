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
    TestUtils.assert_equal(model.bias_vector(q, q_dot), [1058.313451, -6.679008], expand=False)
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
        n_shooting=10,
        final_time=0.5,
        expand_dynamics=False,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(
        states["q_u"],
        np.array(
            [
                [
                    0.00000000e00,
                    -2.42756052e-04,
                    -5.46176583e-03,
                    -2.21847781e-02,
                    -4.21321813e-02,
                    -4.84198776e-02,
                    -5.50960721e-02,
                    -8.34624676e-02,
                    -1.27760701e-01,
                    -1.66731157e-01,
                    -1.77784649e-01,
                    -1.89079118e-01,
                    -2.33428582e-01,
                    -2.96177764e-01,
                    -3.48049081e-01,
                    -3.62405284e-01,
                    -3.76968315e-01,
                    -4.33289684e-01,
                    -5.10056035e-01,
                    -5.70526066e-01,
                    -5.86751923e-01,
                    -6.02969155e-01,
                    -6.63239320e-01,
                    -7.39089606e-01,
                    -7.93706958e-01,
                    -8.07584321e-01,
                    -8.21128855e-01,
                    -8.68579609e-01,
                    -9.21245035e-01,
                    -9.53440226e-01,
                    -9.60711716e-01,
                    -9.67408939e-01,
                    -9.87224385e-01,
                    -9.99865845e-01,
                    -9.98922967e-01,
                    -9.97065237e-01,
                    -9.94554455e-01,
                    -9.79770993e-01,
                    -9.48161004e-01,
                    -9.14869882e-01,
                    -9.04713314e-01,
                    -8.94039400e-01,
                    -8.49571525e-01,
                    -7.81682765e-01,
                    -7.22849648e-01,
                    -7.06285078e-01,
                    -6.89327493e-01,
                    -6.21902196e-01,
                    -5.26640979e-01,
                    -4.49650571e-01,
                    -4.28730524e-01,
                ],
                [
                    1.57079633e00,
                    1.57137136e00,
                    1.58369281e00,
                    1.62294998e00,
                    1.66946377e00,
                    1.68406339e00,
                    1.69943962e00,
                    1.76273318e00,
                    1.85667092e00,
                    1.93528548e00,
                    1.95693953e00,
                    1.97846427e00,
                    2.05477093e00,
                    2.14323978e00,
                    2.20200875e00,
                    2.21627410e00,
                    2.22987552e00,
                    2.27437500e00,
                    2.31664308e00,
                    2.33669175e00,
                    2.34019795e00,
                    2.34292864e00,
                    2.34624788e00,
                    2.33405713e00,
                    2.31200582e00,
                    2.30426711e00,
                    2.29575134e00,
                    2.25692315e00,
                    2.18999079e00,
                    2.12617833e00,
                    2.10733002e00,
                    2.08772018e00,
                    2.00744476e00,
                    1.88645489e00,
                    1.78057455e00,
                    1.75032391e00,
                    1.71940166e00,
                    1.59947266e00,
                    1.43309869e00,
                    1.29735085e00,
                    1.25990646e00,
                    1.22196181e00,
                    1.07529692e00,
                    8.74236824e-01,
                    7.13697867e-01,
                    6.70161269e-01,
                    6.26186086e-01,
                    4.56017881e-01,
                    2.26017379e-01,
                    4.75324708e-02,
                    7.24258406e-05,
                ],
                [
                    0.00000000e00,
                    -4.27196851e-04,
                    -9.60627158e-03,
                    -3.92506859e-02,
                    -7.49842144e-02,
                    -8.63260975e-02,
                    -9.84086081e-02,
                    -1.50134342e-01,
                    -2.31553753e-01,
                    -3.03208060e-01,
                    -3.23466407e-01,
                    -3.44136122e-01,
                    -4.24810117e-01,
                    -5.36131832e-01,
                    -6.24817435e-01,
                    -6.48793311e-01,
                    -6.72870698e-01,
                    -7.63799235e-01,
                    -8.83041942e-01,
                    -9.74505769e-01,
                    -9.98879953e-01,
                    -1.02323493e00,
                    -1.11429676e00,
                    -1.23306215e00,
                    -1.32486373e00,
                    -1.34953595e00,
                    -1.37429135e00,
                    -1.46791512e00,
                    -1.59219121e00,
                    -1.68924236e00,
                    -1.71535763e00,
                    -1.74154439e00,
                    -1.84011621e00,
                    -1.96904640e00,
                    -2.06771744e00,
                    -2.09393944e00,
                    -2.12002468e00,
                    -2.21589458e00,
                    -2.33671140e00,
                    -2.42662283e00,
                    -2.45025744e00,
                    -2.47375316e00,
                    -2.56077280e00,
                    -2.67266785e00,
                    -2.75858852e00,
                    -2.78170366e00,
                    -2.80493688e00,
                    -2.89330543e00,
                    -3.01414149e00,
                    -3.11372577e00,
                    -3.14159265e00,
                ],
            ]
        ),
        decimal=6,
    )


def test_example_arm26_pendulum_swingup_muscle_algebraic():
    """Test the holonomic_constraints/two_pendulums example"""
    from bioptim.examples.toy_examples.holonomic_constraints import arm26_pendulum_swingup_muscle_algebraic

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Prepare the ocp --- #
    ocp, model = arm26_pendulum_swingup_muscle_algebraic.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/arm26_w_pendulum.bioMod",
        n_shooting=10,
        final_time=0.5,
        expand_dynamics=False,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(
        states["q_u"],
        np.array(
            [
                [
                    0.00000000e00,
                    -2.42756110e-04,
                    -5.46176715e-03,
                    -2.21847832e-02,
                    -4.21321905e-02,
                    -4.84198880e-02,
                    -5.50960836e-02,
                    -8.34624820e-02,
                    -1.27760716e-01,
                    -1.66731170e-01,
                    -1.77784661e-01,
                    -1.89079129e-01,
                    -2.33428588e-01,
                    -2.96177762e-01,
                    -3.48049071e-01,
                    -3.62405272e-01,
                    -3.76968301e-01,
                    -4.33289661e-01,
                    -5.10056000e-01,
                    -5.70526023e-01,
                    -5.86751877e-01,
                    -6.02969107e-01,
                    -6.63239264e-01,
                    -7.39089543e-01,
                    -7.93706893e-01,
                    -8.07584255e-01,
                    -8.21128789e-01,
                    -8.68579545e-01,
                    -9.21244978e-01,
                    -9.53440179e-01,
                    -9.60711673e-01,
                    -9.67408899e-01,
                    -9.87224361e-01,
                    -9.99865847e-01,
                    -9.98922993e-01,
                    -9.97065270e-01,
                    -9.94554495e-01,
                    -9.79771058e-01,
                    -9.48161094e-01,
                    -9.14869985e-01,
                    -9.04713420e-01,
                    -8.94039507e-01,
                    -8.49571634e-01,
                    -7.81682865e-01,
                    -7.22849731e-01,
                    -7.06285155e-01,
                    -6.89327563e-01,
                    -6.21902232e-01,
                    -5.26640956e-01,
                    -4.49650496e-01,
                    -4.28730435e-01,
                ],
                [
                    1.57079633e00,
                    1.57137136e00,
                    1.58369281e00,
                    1.62295000e00,
                    1.66946382e00,
                    1.68406345e00,
                    1.69943968e00,
                    1.76273327e00,
                    1.85667104e00,
                    1.93528561e00,
                    1.95693966e00,
                    1.97846441e00,
                    2.05477108e00,
                    2.14323995e00,
                    2.20200892e00,
                    2.21627428e00,
                    2.22987570e00,
                    2.27437518e00,
                    2.31664328e00,
                    2.33669197e00,
                    2.34019818e00,
                    2.34292887e00,
                    2.34624813e00,
                    2.33405741e00,
                    2.31200614e00,
                    2.30426743e00,
                    2.29575167e00,
                    2.25692351e00,
                    2.18999120e00,
                    2.12617878e00,
                    2.10733049e00,
                    2.08772066e00,
                    2.00744528e00,
                    1.88645547e00,
                    1.78057519e00,
                    1.75032457e00,
                    1.71940233e00,
                    1.59947338e00,
                    1.43309944e00,
                    1.29735160e00,
                    1.25990721e00,
                    1.22196255e00,
                    1.07529763e00,
                    8.74237477e-01,
                    7.13698444e-01,
                    6.70161821e-01,
                    6.26186611e-01,
                    4.56018288e-01,
                    2.26017593e-01,
                    4.75325166e-02,
                    7.24259565e-05,
                ],
                [
                    0.00000000e00,
                    -4.27196950e-04,
                    -9.60627378e-03,
                    -3.92506948e-02,
                    -7.49842313e-02,
                    -8.63261170e-02,
                    -9.84086301e-02,
                    -1.50134373e-01,
                    -2.31553795e-01,
                    -3.03208108e-01,
                    -3.23466456e-01,
                    -3.44136172e-01,
                    -4.24810172e-01,
                    -5.36131892e-01,
                    -6.24817496e-01,
                    -6.48793372e-01,
                    -6.72870760e-01,
                    -7.63799297e-01,
                    -8.83042001e-01,
                    -9.74505824e-01,
                    -9.98880006e-01,
                    -1.02323498e00,
                    -1.11429680e00,
                    -1.23306218e00,
                    -1.32486375e00,
                    -1.34953597e00,
                    -1.37429136e00,
                    -1.46791512e00,
                    -1.59219119e00,
                    -1.68924232e00,
                    -1.71535759e00,
                    -1.74154434e00,
                    -1.84011615e00,
                    -1.96904633e00,
                    -2.06771737e00,
                    -2.09393936e00,
                    -2.12002461e00,
                    -2.21589452e00,
                    -2.33671134e00,
                    -2.42662279e00,
                    -2.45025741e00,
                    -2.47375313e00,
                    -2.56077279e00,
                    -2.67266785e00,
                    -2.75858853e00,
                    -2.78170367e00,
                    -2.80493689e00,
                    -2.89330544e00,
                    -3.01414149e00,
                    -3.11372577e00,
                    -3.14159265e00,
                ],
            ]
        ),
        decimal=6,
    )
