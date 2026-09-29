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


def test_example_alignment_constraint():
    """Test the holonomic_constraints/frame_alignment example"""
    from bioptim.examples.toy_examples.holonomic_constraints import frame_alignment_orientation

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Prepare the ocp --- #
    ocp, bio_model = frame_alignment_orientation.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/two_cubes_lagrange2D_outofplane.bioMod",
        n_shooting=10,
        final_time=2.0,
        expand_dynamics=False,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    q, _, _, _ = frame_alignment_orientation.compute_all_states(sol, bio_model)

    print(q)

    npt.assert_almost_equal(
        q,
        np.array(
            [
                [0.0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0],
                [
                    0.44879895,
                    0.45819223,
                    0.48660965,
                    0.53479017,
                    0.60404417,
                    0.69633279,
                    0.81427079,
                    0.9608684,
                    1.1386939,
                    1.34814259,
                    1.58512334,
                ],
                [
                    -0.39269908,
                    -0.39768345,
                    -0.41244445,
                    -0.43637074,
                    -0.46832351,
                    -0.50644603,
                    -0.54788349,
                    -0.58842906,
                    -0.62219113,
                    -0.64154838,
                    -0.6378551,
                ],
                [
                    -0.52359878,
                    -0.51998247,
                    -0.50878286,
                    -0.48893843,
                    -0.45865539,
                    -0.41540327,
                    -0.35600627,
                    -0.27700848,
                    -0.17562772,
                    -0.05161501,
                    0.09027961,
                ],
            ]
        ),
        decimal=6,
    )


def test_example_generalized_alignment_constraint():
    """Test the holonomic_constraints/frame_alignment example"""
    from bioptim.examples.toy_examples.holonomic_constraints import frame_alignment_orientation_6DOF

    bioptim_folder = TestUtils.bioptim_folder()

    n_shooting = 10

    interpolated_points = frame_alignment_orientation_6DOF.build_dummy_trajectory_for_the_driving_cube(n_shooting)

    # --- Prepare the ocp --- #
    ocp = frame_alignment_orientation_6DOF.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/two_cubes_lagrange2D_6DOF.bioMod",
        n_shooting=n_shooting,
        final_time=1.0,
        interpolated_points=interpolated_points,
        expand_dynamics=False,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    stepwise_q_u = sol.stepwise_states(to_merge=SolutionMerge.NODES)["q_u"]
    stepwise_q_v = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)["q_v"]
    q = ocp.nlp[0].model.state_from_partition(stepwise_q_u, stepwise_q_v).toarray()

    npt.assert_almost_equal(
        q,
        np.array(
            [
                [
                    0.00000000e00,
                    1.84183688e-02,
                    8.69533563e-02,
                    1.74878304e-01,
                    2.41034127e-01,
                    2.58471333e-01,
                    2.75800263e-01,
                    3.39618878e-01,
                    4.19924054e-01,
                    4.79143834e-01,
                    4.94576090e-01,
                    5.09804824e-01,
                    5.64631094e-01,
                    6.30606639e-01,
                    6.76895124e-01,
                    6.88601731e-01,
                    6.99962889e-01,
                    7.38794655e-01,
                    7.80446328e-01,
                    8.05503063e-01,
                    8.11180516e-01,
                    8.16375325e-01,
                    8.31013400e-01,
                    8.38669494e-01,
                    8.35872495e-01,
                    8.33872112e-01,
                    8.31360572e-01,
                    8.17528387e-01,
                    7.89167394e-01,
                    7.59682447e-01,
                    7.50710849e-01,
                    7.41351426e-01,
                    7.03492484e-01,
                    6.47783856e-01,
                    6.00417441e-01,
                    5.87131933e-01,
                    5.73646647e-01,
                    5.21962799e-01,
                    4.52113357e-01,
                    3.96840261e-01,
                    3.81869496e-01,
                    3.66848273e-01,
                    3.10481533e-01,
                    2.37007346e-01,
                    1.80774370e-01,
                    1.65803663e-01,
                    1.50855044e-01,
                    9.50955518e-02,
                    2.31169400e-02,
                    -3.15199368e-02,
                    -4.60095466e-02,
                ],
                [
                    0.00000000e00,
                    1.52195502e-02,
                    7.04221945e-02,
                    1.37873836e-01,
                    1.86034800e-01,
                    1.98344622e-01,
                    2.10434143e-01,
                    2.53837566e-01,
                    3.05728932e-01,
                    3.41798442e-01,
                    3.50858164e-01,
                    3.59685678e-01,
                    3.90738731e-01,
                    4.26220292e-01,
                    4.49451401e-01,
                    4.55048672e-01,
                    4.60395149e-01,
                    4.78207858e-01,
                    4.95971679e-01,
                    5.05265803e-01,
                    5.07095915e-01,
                    5.08649722e-01,
                    5.11975699e-01,
                    5.10242071e-01,
                    5.04145384e-01,
                    5.01810932e-01,
                    4.99171598e-01,
                    4.86492511e-01,
                    4.63277268e-01,
                    4.40300335e-01,
                    4.33412990e-01,
                    4.26201211e-01,
                    3.96224192e-01,
                    3.50200390e-01,
                    3.09647271e-01,
                    2.98073348e-01,
                    2.86177374e-01,
                    2.38675918e-01,
                    1.70011949e-01,
                    1.12365469e-01,
                    9.62843392e-02,
                    7.99033422e-02,
                    1.57853353e-02,
                    -7.39840143e-02,
                    -1.47313278e-01,
                    -1.67495822e-01,
                    -1.87944551e-01,
                    -2.67020368e-01,
                    -3.75553741e-01,
                    -4.62678911e-01,
                    -4.86451673e-01,
                ],
                [
                    0.00000000e00,
                    2.28995880e-02,
                    1.05223199e-01,
                    2.04086355e-01,
                    2.73356394e-01,
                    2.90866995e-01,
                    3.07979846e-01,
                    3.68662489e-01,
                    4.39479516e-01,
                    4.87403344e-01,
                    4.99248313e-01,
                    5.10705293e-01,
                    5.50252878e-01,
                    5.93711667e-01,
                    6.20831039e-01,
                    6.27156119e-01,
                    6.33103712e-01,
                    6.52074381e-01,
                    6.68920270e-01,
                    6.75817206e-01,
                    6.76778831e-01,
                    6.77374223e-01,
                    6.76364689e-01,
                    6.67381642e-01,
                    6.54650300e-01,
                    6.50404528e-01,
                    6.45801264e-01,
                    6.25350257e-01,
                    5.91134344e-01,
                    5.59132073e-01,
                    5.49757523e-01,
                    5.40024689e-01,
                    5.00282566e-01,
                    4.40746180e-01,
                    3.89165837e-01,
                    3.74543230e-01,
                    3.59546502e-01,
                    2.99885050e-01,
                    2.13923433e-01,
                    1.41715600e-01,
                    1.21538179e-01,
                    1.00961888e-01,
                    2.01412539e-02,
                    -9.39556377e-02,
                    -1.88133332e-01,
                    -2.14223310e-01,
                    -2.40735390e-01,
                    -3.44025644e-01,
                    -4.87869855e-01,
                    -6.05145894e-01,
                    -6.37430021e-01,
                ],
                [
                    0.00000000e00,
                    1.11247018e-02,
                    4.95949696e-02,
                    9.15952168e-02,
                    1.17133856e-01,
                    1.22924936e-01,
                    1.28281102e-01,
                    1.44433299e-01,
                    1.55696423e-01,
                    1.56308418e-01,
                    1.55231633e-01,
                    1.53622566e-01,
                    1.42764836e-01,
                    1.16396507e-01,
                    8.60992059e-02,
                    7.64672370e-02,
                    6.61757230e-02,
                    2.17188925e-02,
                    -5.04205118e-02,
                    -1.16228013e-01,
                    -1.35197885e-01,
                    -1.54725263e-01,
                    -2.32668171e-01,
                    -3.42537649e-01,
                    -4.29543109e-01,
                    -4.52668157e-01,
                    -4.75687410e-01,
                    -5.60372089e-01,
                    -6.63128265e-01,
                    -7.33163230e-01,
                    -7.50323623e-01,
                    -7.66820206e-01,
                    -8.22515605e-01,
                    -8.80275768e-01,
                    -9.13374057e-01,
                    -9.20631714e-01,
                    -9.27238321e-01,
                    -9.46365050e-01,
                    -9.58707999e-01,
                    -9.59290431e-01,
                    -9.58230580e-01,
                    -9.56674384e-01,
                    -9.46551252e-01,
                    -9.23774036e-01,
                    -8.99538966e-01,
                    -8.92148710e-01,
                    -8.84380924e-01,
                    -8.51996152e-01,
                    -8.02637583e-01,
                    -7.59987364e-01,
                    -7.47996280e-01,
                ],
                [
                    0.00000000e00,
                    1.16335926e-02,
                    5.80615655e-02,
                    1.24340089e-01,
                    1.78645194e-01,
                    1.93528583e-01,
                    2.08558685e-01,
                    2.66024627e-01,
                    3.42608257e-01,
                    4.01608440e-01,
                    4.17263645e-01,
                    4.32870984e-01,
                    4.90835824e-01,
                    5.63921189e-01,
                    6.16812675e-01,
                    6.30303331e-01,
                    6.43521306e-01,
                    6.90557261e-01,
                    7.44376588e-01,
                    7.78406571e-01,
                    7.86252436e-01,
                    7.93566493e-01,
                    8.16141062e-01,
                    8.32894962e-01,
                    8.35297932e-01,
                    8.34375595e-01,
                    8.32802073e-01,
                    8.21117387e-01,
                    7.92846696e-01,
                    7.62206000e-01,
                    7.52879998e-01,
                    7.43090178e-01,
                    7.02490129e-01,
                    6.41944597e-01,
                    5.91237234e-01,
                    5.77260194e-01,
                    5.63113410e-01,
                    5.08755403e-01,
                    4.36096059e-01,
                    3.80235363e-01,
                    3.65448277e-01,
                    3.50724713e-01,
                    2.96222827e-01,
                    2.27870112e-01,
                    1.78503603e-01,
                    1.65891310e-01,
                    1.53528837e-01,
                    1.09541911e-01,
                    5.86720026e-02,
                    2.54147046e-02,
                    1.74698811e-02,
                ],
                [
                    0.00000000e00,
                    3.02476726e-02,
                    1.42759272e-01,
                    2.87649334e-01,
                    3.97858767e-01,
                    4.27187863e-01,
                    4.56525716e-01,
                    5.66944669e-01,
                    7.12785660e-01,
                    8.27038273e-01,
                    8.57981367e-01,
                    8.89162409e-01,
                    1.00853310e00,
                    1.17123921e00,
                    1.30261670e00,
                    1.33874869e00,
                    1.37537211e00,
                    1.51727562e00,
                    1.71383379e00,
                    1.87344158e00,
                    1.91723486e00,
                    1.96152778e00,
                    2.13201892e00,
                    2.36218913e00,
                    2.54153889e00,
                    2.58934484e00,
                    2.63708899e00,
                    2.81506775e00,
                    3.04123578e00,
                    3.20765421e00,
                    3.25081714e00,
                    3.29346046e00,
                    3.44865369e00,
                    3.64002609e00,
                    3.77898419e00,
                    3.81501703e00,
                    3.85064739e00,
                    3.98099869e00,
                    4.14448579e00,
                    4.26607470e00,
                    4.29807307e00,
                    4.32992813e00,
                    4.44842218e00,
                    4.60154819e00,
                    4.71866864e00,
                    4.74992639e00,
                    4.78122211e00,
                    4.89913969e00,
                    5.05452899e00,
                    5.17508073e00,
                    5.20742850e00,
                ],
                [
                    0.00000000e00,
                    1.30038173e-02,
                    6.18352909e-02,
                    1.25509622e-01,
                    1.74180072e-01,
                    1.87117015e-01,
                    2.00017772e-01,
                    2.47906771e-01,
                    3.09029378e-01,
                    3.54733597e-01,
                    3.66733165e-01,
                    3.78611106e-01,
                    4.21688213e-01,
                    4.74234839e-01,
                    5.11623640e-01,
                    5.21155231e-01,
                    5.30439034e-01,
                    5.62477091e-01,
                    5.97561898e-01,
                    6.19263853e-01,
                    6.24283805e-01,
                    6.28930106e-01,
                    6.42564809e-01,
                    6.51327306e-01,
                    6.51145190e-01,
                    6.50091087e-01,
                    6.48623976e-01,
                    6.39526088e-01,
                    6.19204187e-01,
                    5.97245968e-01,
                    5.90473711e-01,
                    5.83378394e-01,
                    5.54448411e-01,
                    5.11399126e-01,
                    4.74506953e-01,
                    4.64126550e-01,
                    4.53580706e-01,
                    4.13116935e-01,
                    3.58407645e-01,
                    3.15187007e-01,
                    3.03501902e-01,
                    2.91790546e-01,
                    2.48001401e-01,
                    1.91409431e-01,
                    1.48583791e-01,
                    1.37266763e-01,
                    1.26005979e-01,
                    8.43918380e-02,
                    3.17124394e-02,
                    -7.36968042e-03,
                    -1.75898471e-02,
                ],
                [
                    -2.00000000e00,
                    -2.00951339e00,
                    -2.04475267e00,
                    -2.08960446e00,
                    -2.12310700e00,
                    -2.13190588e00,
                    -2.14063458e00,
                    -2.17262301e00,
                    -2.21255334e00,
                    -2.24180783e00,
                    -2.24941021e00,
                    -2.25689803e00,
                    -2.28368156e00,
                    -2.31557096e00,
                    -2.33775253e00,
                    -2.34334232e00,
                    -2.34875036e00,
                    -2.36701203e00,
                    -2.38613186e00,
                    -2.39730699e00,
                    -2.39979042e00,
                    -2.40203288e00,
                    -2.40800830e00,
                    -2.41016600e00,
                    -2.40765123e00,
                    -2.40638818e00,
                    -2.40489063e00,
                    -2.39731292e00,
                    -2.38292954e00,
                    -2.36859941e00,
                    -2.36431208e00,
                    -2.35987685e00,
                    -2.34234841e00,
                    -2.31746611e00,
                    -2.29691625e00,
                    -2.29122921e00,
                    -2.28549802e00,
                    -2.26398771e00,
                    -2.23589476e00,
                    -2.21429567e00,
                    -2.20852305e00,
                    -2.20276962e00,
                    -2.18158033e00,
                    -2.15477062e00,
                    -2.13472904e00,
                    -2.12944604e00,
                    -2.12419255e00,
                    -2.10478626e00,
                    -2.08002401e00,
                    -2.06129091e00,
                    -2.05631410e00,
                ],
                [
                    0.00000000e00,
                    2.89077537e-02,
                    1.33287992e-01,
                    2.59732999e-01,
                    3.49193478e-01,
                    3.71940438e-01,
                    3.94226717e-01,
                    4.73740491e-01,
                    5.67688787e-01,
                    6.32186728e-01,
                    6.48272450e-01,
                    6.63891313e-01,
                    7.18308488e-01,
                    7.79319124e-01,
                    8.18406519e-01,
                    8.27695422e-01,
                    8.36504431e-01,
                    8.65230021e-01,
                    8.92390424e-01,
                    9.05271810e-01,
                    9.07538997e-01,
                    9.09310471e-01,
                    9.11488494e-01,
                    9.03685973e-01,
                    8.89518312e-01,
                    8.84541230e-01,
                    8.79057720e-01,
                    8.53955149e-01,
                    8.10433189e-01,
                    7.68792694e-01,
                    7.56481620e-01,
                    7.43662811e-01,
                    6.91058938e-01,
                    6.11725131e-01,
                    5.42693722e-01,
                    5.23093063e-01,
                    5.02988148e-01,
                    4.23068895e-01,
                    3.08174314e-01,
                    2.11957585e-01,
                    1.85124957e-01,
                    1.57791822e-01,
                    5.07692654e-02,
                    -9.94095665e-02,
                    -2.22588566e-01,
                    -2.56590902e-01,
                    -2.91088696e-01,
                    -4.24964166e-01,
                    -6.10067603e-01,
                    -7.59905222e-01,
                    -8.00992172e-01,
                ],
                [
                    4.48798951e-01,
                    4.65491412e-01,
                    5.25852161e-01,
                    5.97809403e-01,
                    6.46685808e-01,
                    6.58702865e-01,
                    6.70281360e-01,
                    7.09735915e-01,
                    7.51449622e-01,
                    7.75788792e-01,
                    7.81149213e-01,
                    7.86045248e-01,
                    8.00388927e-01,
                    8.09482039e-01,
                    8.09028841e-01,
                    8.07809756e-01,
                    8.06143872e-01,
                    7.96089590e-01,
                    7.73733773e-01,
                    7.49380387e-01,
                    7.41811965e-01,
                    7.33797946e-01,
                    6.99695242e-01,
                    6.45626056e-01,
                    5.96944715e-01,
                    5.82956944e-01,
                    5.68539957e-01,
                    5.10614622e-01,
                    4.27142232e-01,
                    3.58645484e-01,
                    3.39965578e-01,
                    3.21154829e-01,
                    2.50008201e-01,
                    1.58526180e-01,
                    9.19902716e-02,
                    7.50406978e-02,
                    5.84751389e-02,
                    1.33716754e-04,
                    -6.55962259e-02,
                    -1.07198490e-01,
                    -1.16954684e-01,
                    -1.26148963e-01,
                    -1.55678559e-01,
                    -1.82549514e-01,
                    -1.94484710e-01,
                    -1.96434461e-01,
                    -1.97877817e-01,
                    -1.98882293e-01,
                    -1.90054127e-01,
                    -1.75849311e-01,
                    -1.71011478e-01,
                ],
                [
                    -3.92699082e-01,
                    -3.88136818e-01,
                    -3.66126749e-01,
                    -3.26610549e-01,
                    -2.88967908e-01,
                    -2.77977358e-01,
                    -2.66615807e-01,
                    -2.20974972e-01,
                    -1.55385463e-01,
                    -1.01544750e-01,
                    -8.67941019e-02,
                    -7.18972015e-02,
                    -1.48966783e-02,
                    6.11137254e-02,
                    1.19683566e-01,
                    1.35229448e-01,
                    1.50735822e-01,
                    2.08474718e-01,
                    2.81809639e-01,
                    3.35548770e-01,
                    3.49396637e-01,
                    3.63035380e-01,
                    4.12342992e-01,
                    4.71105675e-01,
                    5.10841986e-01,
                    5.20524860e-01,
                    5.29815491e-01,
                    5.60985225e-01,
                    5.91820229e-01,
                    6.07008160e-01,
                    6.09737819e-01,
                    6.11889027e-01,
                    6.14588855e-01,
                    6.05369612e-01,
                    5.88846687e-01,
                    5.83138517e-01,
                    5.76888375e-01,
                    5.48764017e-01,
                    5.02199148e-01,
                    4.60233090e-01,
                    4.48284144e-01,
                    4.36042731e-01,
                    3.87766944e-01,
                    3.20528727e-01,
                    2.66957493e-01,
                    2.52516897e-01,
                    2.38036497e-01,
                    1.83589815e-01,
                    1.13375225e-01,
                    6.12071303e-02,
                    4.76607485e-02,
                ],
                [
                    -5.23598776e-01,
                    -4.86933516e-01,
                    -3.50845716e-01,
                    -1.77685387e-01,
                    -4.88076633e-02,
                    -1.50459449e-02,
                    1.84755751e-02,
                    1.42220569e-01,
                    2.99247702e-01,
                    4.16842344e-01,
                    4.47860822e-01,
                    4.78766376e-01,
                    5.93919244e-01,
                    7.43135273e-01,
                    8.57681423e-01,
                    8.88343859e-01,
                    9.19083971e-01,
                    1.03527017e00,
                    1.18979889e00,
                    1.31139056e00,
                    1.34435271e00,
                    1.37757005e00,
                    1.50476572e00,
                    1.67735635e00,
                    1.81520423e00,
                    1.85276781e00,
                    1.89069761e00,
                    2.03639996e00,
                    2.23383590e00,
                    2.38976199e00,
                    2.43180948e00,
                    2.47402099e00,
                    2.63331774e00,
                    2.84112961e00,
                    2.99836183e00,
                    3.03976519e00,
                    3.08090797e00,
                    3.23262761e00,
                    3.42348423e00,
                    3.56410421e00,
                    3.60076974e00,
                    3.63710453e00,
                    3.77064406e00,
                    3.93899874e00,
                    4.06446831e00,
                    4.09749315e00,
                    4.13037472e00,
                    4.25275144e00,
                    4.41098259e00,
                    4.53208865e00,
                    4.56443031e00,
                ],
            ]
        ),
        decimal=6,
    )
