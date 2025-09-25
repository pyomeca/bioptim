import platform

import numpy as np
import numpy.testing as npt
import pytest
from casadi import DM, MX

from bioptim import (
    HolonomicBiorbdModel,
    HolonomicConstraintsFcn,
    HolonomicConstraintsList,
    OdeSolver,
    SolutionMerge,
    Solver,
)

from ..utils import TestUtils


def test_model_holonomic():
    from bioptim.examples.torque_driven_ocp import example_multi_biorbd_model as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)
    biorbd_model_path = bioptim_folder + "/models/triple_pendulum.bioMod"
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
    from bioptim.examples.holonomic_constraints import two_pendulums

    bioptim_folder = TestUtils.module_folder(two_pendulums)

    # --- Prepare the ocp --- #
    ocp, model = two_pendulums.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/two_pendulums.bioMod",
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
                    1.53993574,
                    1.53854879,
                    1.53402211,
                    1.52847541,
                    1.52669413,
                    1.52479253,
                    1.51664893,
                    1.50364381,
                    1.49186773,
                    1.48846757,
                    1.48496766,
                    1.47103762,
                    1.45101122,
                    1.43428871,
                    1.4296383,
                    1.42491613,
                    1.40664125,
                    1.38154687,
                    1.3614185,
                    1.35593186,
                    1.35040012,
                    1.32929443,
                    1.30098709,
                    1.27874555,
                    1.27274408,
                    1.26671486,
                    1.24387107,
                    1.21357331,
                    1.18999023,
                    1.18365498,
                    1.17730007,
                    1.15329038,
                    1.12158414,
                    1.0969888,
                    1.09039172,
                    1.08377726,
                    1.05880536,
                    1.02586166,
                    1.00032296,
                    0.99347443,
                    0.98660795,
                    0.96068097,
                    0.92646613,
                    0.89993169,
                    0.89281453,
                    0.88567777,
                    0.85871999,
                    0.82312202,
                    0.79549941,
                    0.78808837,
                    0.78065602,
                    0.75257307,
                    0.71547266,
                    0.68667555,
                    0.67894863,
                    0.67119919,
                    0.64191606,
                    0.60323049,
                    0.57320863,
                    0.56515452,
                    0.55707768,
                    0.52656501,
                    0.48627929,
                    0.45504017,
                    0.44666377,
                    0.43826568,
                    0.40655829,
                    0.36474618,
                    0.33236827,
                    0.3236937,
                    0.31499985,
                    0.28220556,
                    0.23903604,
                    0.20566982,
                    0.19674014,
                    0.18779476,
                    0.1540893,
                    0.10981354,
                    0.07566688,
                    0.06653957,
                    0.05740078,
                    0.02300652,
                    -0.02207661,
                    -0.05676907,
                    -0.06603084,
                    -0.07529994,
                    -0.11015022,
                    -0.15574627,
                    -0.19076596,
                    -0.20010498,
                    -0.20944811,
                    -0.24455526,
                    -0.29043121,
                    -0.32561834,
                    -0.3349946,
                    -0.34437316,
                    -0.37960761,
                    -0.42562773,
                    -0.46090085,
                    -0.47029563,
                    -0.47969201,
                    -0.51499618,
                    -0.56110295,
                    -0.59642867,
                    -0.60583432,
                    -0.61524053,
                    -0.65057462,
                    -0.6966921,
                    -0.73199308,
                    -0.74138606,
                    -0.75077582,
                    -0.78600437,
                    -0.83187235,
                    -0.86688888,
                    -0.87619157,
                    -0.88548196,
                    -0.92023137,
                    -0.9652234,
                    -0.99937677,
                    -1.00842136,
                    -1.01743808,
                    -1.05099004,
                    -1.09401872,
                    -1.12636044,
                    -1.13487707,
                    -1.14334488,
                    -1.17463259,
                    -1.21420857,
                    -1.243506,
                    -1.25115111,
                    -1.2587228,
                    -1.28643644,
                    -1.32083863,
                    -1.34578378,
                    -1.35221472,
                    -1.35855226,
                    -1.38148194,
                    -1.40935091,
                    -1.42914168,
                    -1.43418725,
                    -1.43913685,
                    -1.45685819,
                    -1.47799261,
                    -1.49270762,
                    -1.49641559,
                    -1.50003391,
                    -1.51281478,
                    -1.52762174,
                    -1.53755795,
                    -1.54,
                ],
                [
                    1.54,
                    1.54008014,
                    1.54180759,
                    1.54746234,
                    1.55441753,
                    1.55665635,
                    1.55904689,
                    1.56927654,
                    1.58558391,
                    1.60029652,
                    1.60453126,
                    1.60888033,
                    1.62606456,
                    1.65040709,
                    1.67039788,
                    1.67590201,
                    1.68146392,
                    1.70270917,
                    1.73117772,
                    1.75344302,
                    1.75942645,
                    1.76542049,
                    1.78792093,
                    1.81720155,
                    1.83950479,
                    1.84541875,
                    1.8513145,
                    1.87322903,
                    1.9012715,
                    1.9223014,
                    1.92783272,
                    1.93333025,
                    1.95363296,
                    1.97931665,
                    1.9983619,
                    2.00334038,
                    2.00827631,
                    2.02640436,
                    2.04909814,
                    2.0657394,
                    2.07006118,
                    2.07433428,
                    2.08992502,
                    2.10918766,
                    2.12310322,
                    2.1266842,
                    2.13021066,
                    2.14294933,
                    2.15836564,
                    2.16923076,
                    2.1719832,
                    2.1746746,
                    2.18422193,
                    2.19533145,
                    2.20277982,
                    2.20460426,
                    2.20636031,
                    2.21233241,
                    2.21861873,
                    2.22224541,
                    2.22303296,
                    2.22374442,
                    2.22572876,
                    2.2266518,
                    2.2260458,
                    2.2256879,
                    2.22524674,
                    2.22284361,
                    2.21790062,
                    2.21269591,
                    2.21109843,
                    2.20941206,
                    2.20228899,
                    2.19108649,
                    2.18101696,
                    2.17811468,
                    2.17512034,
                    2.16306384,
                    2.14537972,
                    2.1303221,
                    2.12608947,
                    2.1217645,
                    2.10471399,
                    2.08053006,
                    2.06051993,
                    2.05497399,
                    2.04933816,
                    2.02738735,
                    1.99687372,
                    1.9720795,
                    1.96527088,
                    1.95837641,
                    1.9317282,
                    1.89516443,
                    1.86581045,
                    1.85780023,
                    1.849708,
                    1.81858126,
                    1.77622598,
                    1.74248584,
                    1.73331626,
                    1.72406663,
                    1.68859438,
                    1.64057373,
                    1.60250505,
                    1.59218549,
                    1.58178627,
                    1.54199289,
                    1.48833052,
                    1.44594882,
                    1.43448352,
                    1.42294123,
                    1.37889431,
                    1.31979443,
                    1.27335769,
                    1.26083154,
                    1.24824113,
                    1.2004178,
                    1.13681059,
                    1.08728415,
                    1.07399278,
                    1.06066803,
                    1.0104253,
                    0.94451277,
                    0.89391129,
                    0.88043788,
                    0.86697862,
                    0.81668317,
                    0.75175895,
                    0.70267815,
                    0.68971282,
                    0.67680085,
                    0.62886792,
                    0.5675696,
                    0.52147556,
                    0.50931008,
                    0.49719175,
                    0.45211244,
                    0.39403838,
                    0.34989285,
                    0.33816079,
                    0.3264395,
                    0.28252367,
                    0.2252666,
                    0.18131209,
                    0.16958211,
                    0.15784574,
                    0.11375566,
                    0.05610692,
                    0.01181684,
                    0.0,
                ],
            ],
            decimal=6,
        )


def test_example_two_pendulums_algebraic():
    """Test the holonomic_constraints/two_pendulums_algebraic example"""
    from bioptim.examples.holonomic_constraints import two_pendulums_algebraic

    if platform.system() == "Windows":
        pytest.skip("This test is skipped on Windows because too sensitive.")

    bioptim_folder = TestUtils.module_folder(two_pendulums_algebraic)

    # --- Prepare the ocp --- #
    ocp, model = two_pendulums_algebraic.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/two_pendulums.bioMod",
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
