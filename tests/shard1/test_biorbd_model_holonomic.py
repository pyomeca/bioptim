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
