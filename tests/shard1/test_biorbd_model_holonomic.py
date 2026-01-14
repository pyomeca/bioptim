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
        model.partitioned_forward_dynamics_with_qv()(q_u, q_v[0], qdot_u, tau), [-3.326526], expand=False
    )

    # Test partitioned_forward_dynamics_full
    TestUtils.assert_equal(model.partitioned_forward_dynamics_full()(q, qdot_u, tau), [-23.937828], expand=False)

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
    TestUtils.assert_equal(model.holonomic_constraints_double_derivative(q, q_dot, q_ddot), [10.23374996, -11.73729905])
    TestUtils.assert_equal(
        model.constrained_forward_dynamics()(q, q_dot, tau, []), [-5.18551845, -3.01921376, 25.79451813]
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

    TestUtils.assert_equal(model.partitioned_forward_dynamics()(q_u, qdot_u, q_v, tau), -1.101808, expand=False)
    TestUtils.assert_equal(model.coupling_matrix(q), [5.79509793, -0.35166415], expand=False)
    TestUtils.assert_equal(model.biais_vector(q, q_dot), [27.03137348, 23.97095718], expand=False)
    TestUtils.assert_equal(model.state_from_partition(q_u, q_v), q)

    TestUtils.assert_equal(model.compute_q_v()(q_u, q_v), [2 * np.pi / 3, 2 * np.pi / 3], expand=False)
    TestUtils.assert_equal(model.compute_q()(q_u, q_v), [1.0, 2.0943951, 2.0943951], expand=False)
    TestUtils.assert_equal(model.compute_qdot_v()(q, qdot_u), [23.18039172, -1.4066566], expand=False)
    TestUtils.assert_equal(model.compute_qdot()(q, qdot_u), [4.0, 23.18039172, -1.4066566], expand=False)

    TestUtils.assert_equal(model.compute_qddot_v()(q, q_dot, q_ddot_u), [67.597059, 21.509308], expand=False)
    TestUtils.assert_equal(model.compute_qddot()(q, q_dot, q_ddot_u), [7.0, 67.597059, 21.509308], expand=False)

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
                    1.433706,
                    1.185046,
                    0.891157,
                    0.561607,
                    0.191792,
                    -0.206511,
                    -0.614976,
                    -1.018383,
                    -1.356253,
                    -1.54,
                ],
                [1.54, 1.669722, 1.924726, 2.127746, 2.226937, 2.184007, 1.972105, 1.593534, 1.06751, 0.507334, 0.0],
            ],
            decimal=6,
        )

    elif isinstance(ode_solver, OdeSolver.COLLOCATION):
        npt.assert_almost_equal(
            states["q_u"],
            [
                [
                    1.54,
                    1.53947255,
                    1.52829032,
                    1.4918706,
                    1.44772412,
                    1.43369574,
                    1.41898275,
                    1.35974706,
                    1.27447461,
                    1.20430304,
                    1.18502396,
                    1.16556562,
                    1.09140745,
                    0.99166684,
                    0.91261473,
                    0.8911305,
                    0.86947629,
                    0.78671683,
                    0.67482281,
                    0.585796,
                    0.56157817,
                    0.53715892,
                    0.44378604,
                    0.31793766,
                    0.21860716,
                    0.19175812,
                    0.1647666,
                    0.0623588,
                    -0.07337049,
                    -0.1784637,
                    -0.20655284,
                    -0.23470758,
                    -0.34109507,
                    -0.48052131,
                    -0.58686908,
                    -0.6150277,
                    -0.64319721,
                    -0.74951225,
                    -0.88766222,
                    -0.99132882,
                    -1.01844059,
                    -1.0452361,
                    -1.14223391,
                    -1.25827372,
                    -1.33704454,
                    -1.35629421,
                    -1.3747351,
                    -1.43601269,
                    -1.49800372,
                    -1.53258632,
                    -1.54,
                ],
                [
                    1.54,
                    1.54064515,
                    1.55431106,
                    1.59922772,
                    1.65296364,
                    1.66973502,
                    1.68705873,
                    1.75370598,
                    1.84171006,
                    1.90763103,
                    1.92475754,
                    1.94158775,
                    2.00129161,
                    2.07058481,
                    2.11662288,
                    2.12778938,
                    2.13845817,
                    2.17380886,
                    2.20803169,
                    2.2242531,
                    2.22699357,
                    2.22904554,
                    2.2304308,
                    2.21629085,
                    2.19241948,
                    2.18407247,
                    2.17489014,
                    2.1330352,
                    2.06082354,
                    1.99228721,
                    1.97216828,
                    1.95123087,
                    1.86495558,
                    1.73527252,
                    1.62449462,
                    1.59357668,
                    1.56195524,
                    1.43562583,
                    1.25495762,
                    1.10760291,
                    1.06752078,
                    1.0272699,
                    0.8761086,
                    0.68415659,
                    0.54376452,
                    0.50733629,
                    0.47128493,
                    0.3387245,
                    0.16767637,
                    0.03546255,
                    0.0,
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
    ocp, model = two_pendulums_algebraic.prepare_ocp(
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
                    1.53645293,
                    1.49090493,
                    1.46110856,
                    1.41970745,
                    1.22293077,
                    1.12028136,
                    1.01244361,
                    0.71880381,
                    0.61168231,
                    0.47492593,
                    -0.04877796,
                    -0.29539943,
                    -0.57208567,
                    -1.29065614,
                    -1.54,
                ],
                [
                    1.54,
                    1.53171625,
                    1.43067973,
                    1.3649996,
                    1.29810734,
                    1.10897703,
                    1.0374163,
                    0.93992459,
                    0.52348463,
                    0.31612114,
                    0.12283504,
                    -0.14736347,
                    -0.15187642,
                    -0.15372295,
                    -0.06603706,
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
                    0.99941032,
                    0.99681038,
                    0.99399033,
                    0.98860777,
                    0.94010247,
                    0.90022299,
                    0.84812898,
                    0.6584849,
                    0.57424555,
                    0.45727258,
                    -0.04875862,
                    -0.29112201,
                    -0.5413868,
                    -0.96101669,
                    -0.99952583,
                ],
                [
                    -0.03079146,
                    -0.03433664,
                    -0.07980643,
                    -0.10946795,
                    -0.15051469,
                    -0.34089199,
                    -0.43542918,
                    -0.52978981,
                    -0.75259394,
                    -0.81868312,
                    -0.88932659,
                    -0.99881059,
                    -0.95668593,
                    -0.84077365,
                    -0.27649037,
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
                    -0.27855252,
                    -1.03630477,
                    -1.31246541,
                    -1.95618328,
                    -3.72426958,
                    -4.37488071,
                    -4.24866308,
                    -4.22798705,
                    -4.33906877,
                    -5.97324789,
                    -9.14477984,
                    -9.83232325,
                    -10.76632382,
                    -9.97701724,
                    -8.46520416,
                ],
                [
                    0.0,
                    -0.62989128,
                    -2.28678176,
                    -2.86981849,
                    -2.66247711,
                    -2.79724415,
                    -3.10324185,
                    -4.42493009,
                    -7.59665489,
                    -8.59683036,
                    -6.62468483,
                    -1.17527383,
                    0.84182637,
                    0.28582148,
                    2.24545193,
                    4.23600587,
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
                1.28112296,
                1.22535242,
                1.13876849,
                1.02751913,
                0.89519494,
                0.74384486,
                0.57524378,
                0.39190694,
                0.19799513,
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
                0.76754753,
                0.76031326,
                0.74858082,
                0.73266023,
                0.71288935,
                0.6896361,
                0.66330062,
                0.63432063,
                0.60319444,
                0.5705717,
                0.53750942,
                0.50529314,
                0.47306849,
                0.4387326,
                0.39365688,
                0.33550945,
                0.2727274,
                0.20440837,
                0.13396314,
                0.06433518,
                -0.00455076,
                -0.07445804,
                -0.1471207,
                -0.22344713,
                -0.30363321,
                -0.38765837,
                -0.47567902,
                -0.56818751,
                -0.66598663,
                -0.77,
            ],
            [
                0.0,
                -0.0020824,
                -0.00821695,
                -0.01830149,
                -0.03236938,
                -0.05048038,
                -0.0726346,
                -0.09875238,
                -0.1287542,
                -0.16275482,
                -0.20132215,
                -0.24535886,
                -0.29370214,
                -0.33909243,
                -0.37268925,
                -0.39294823,
                -0.40355002,
                -0.39595389,
                -0.37802695,
                -0.35578337,
                -0.32816605,
                -0.29365283,
                -0.25313732,
                -0.20969424,
                -0.16682135,
                -0.12698646,
                -0.09139836,
                -0.0605062,
                -0.03456109,
                -0.01404837,
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
                -0.47400672,
                -0.39832961,
                -0.28240044,
                -0.14064867,
                0.01147023,
                0.15989866,
                0.29301816,
                0.40093225,
                0.47362475,
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
                0.52359878,
                0.49719372,
                0.41660698,
                0.28429009,
                0.10904561,
                -0.09605531,
                -0.31061947,
                -0.50569474,
                -0.65398428,
                -0.74758078,
                -0.78539816,
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
                -6.23253084e-01,
                -6.38484167e-01,
                -6.68922578e-01,
                -6.81792550e-01,
                -6.55238890e-01,
                -5.87796062e-01,
                -4.89966451e-01,
                -3.76043646e-01,
                -2.59102081e-01,
                -1.48895389e-01,
                -5.20034953e-02,
                2.53932736e-02,
                7.47309323e-02,
                8.76935179e-02,
                6.12325532e-02,
                -1.52507329e-03,
                -9.10955428e-02,
                -1.94140944e-01,
                -2.97712860e-01,
                -3.90212593e-01,
                -4.61042809e-01,
                -5.01160801e-01,
                -5.04847569e-01,
                -4.71389909e-01,
                -4.05573563e-01,
                -3.16705448e-01,
                -2.16476676e-01,
                -1.17080223e-01,
                -3.07197021e-02,
                2.99772919e-02,
                5.23003485e-02,
            ],
            [
                3.80567092e-01,
                4.51818983e-01,
                6.23662754e-01,
                8.05120614e-01,
                9.30553547e-01,
                9.83693526e-01,
                9.77409213e-01,
                9.35719215e-01,
                8.82442686e-01,
                8.35220491e-01,
                8.04531259e-01,
                8.00362705e-01,
                8.40009103e-01,
                9.41814395e-01,
                1.11224555e00,
                1.34321756e00,
                1.61055797e00,
                1.88083666e00,
                2.12537366e00,
                2.32559227e00,
                2.47363237e00,
                2.57049953e00,
                2.62094933e00,
                2.62962518e00,
                2.60031256e00,
                2.53707423e00,
                2.44633497e00,
                2.33899114e00,
                2.23156343e00,
                2.14666431e00,
                2.11304119e00,
            ],
            [
                0.00000000e00,
                2.19071925e-02,
                8.06926144e-02,
                1.66816334e-01,
                2.70564080e-01,
                3.80497065e-01,
                4.82560923e-01,
                5.63694198e-01,
                6.12524398e-01,
                6.20141635e-01,
                5.80996906e-01,
                4.91848034e-01,
                3.50419451e-01,
                1.56747128e-01,
                -8.41634283e-02,
                -3.57931966e-01,
                -6.41571324e-01,
                -9.16060238e-01,
                -1.17294667e00,
                -1.40983578e00,
                -1.62710399e00,
                -1.82869131e00,
                -2.02079309e00,
                -2.20817341e00,
                -2.39205176e00,
                -2.57114387e00,
                -2.74170772e00,
                -2.89615124e00,
                -3.02328610e00,
                -3.11002119e00,
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
