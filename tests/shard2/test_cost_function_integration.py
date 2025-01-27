"""
Test for file IO
"""

import io
import sys

from bioptim import (
    BiorbdModel,
    OdeSolver,
    OdeSolverBase,
    ControlType,
    QuadratureRule,
    OptimalControlProgram,
    Objective,
    ObjectiveFcn,
    Dynamics,
    DynamicsFcn,
    BoundsList,
    Solver,
    PhaseDynamics,
    SolutionMerge,
)
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    integration_rule: QuadratureRule,
    control_type: ControlType,
    objective: str,
    target: np.ndarray = None,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    integration_rule: QuadratureRule
        The integration rule to use
    control_type: ControlType
        The type of control to use (constant or linear)
    objective: str
        The objective to minimize (torque or power)
    target: np.array
        The target value to reach
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    if objective == "torque":
        objective_functions = Objective(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", integration_rule=integration_rule, target=target
        )
    elif objective == "qdot":
        objective_functions = Objective(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", integration_rule=integration_rule, target=target
        )
    elif objective == "mayer":
        objective_functions = Objective(
            ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", integration_rule=integration_rule, target=target
        )
    else:
        raise ValueError("Wrong objective")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, phase_dynamics=phase_dynamics)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 3.14
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max = -100, 100
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau
    u_bounds["tau"][1, :] = 0  # Prevent the model from actively rotate

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        phase_time=1,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        use_sx=True,
        n_threads=1,
        control_type=control_type,
    )


def sum_cost_function_output(sol):
    """
    Sum the cost function output from sol.print_cost()
    """
    captured_output = io.StringIO()  # Create StringIO object
    sys.stdout = captured_output  # and redirect stdout.
    sol.print_cost()  # Call function.
    sys.stdout = sys.__stdout__  # Reset redirect.
    idx = captured_output.getvalue().find("Sum cost functions")
    output = captured_output.getvalue()[idx:].split("\n")[0]
    idx = len("Sum cost functions: ")
    return float(output[idx:])


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("objective", ["torque", "qdot"])
@pytest.mark.parametrize(
    "control_type", [ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE, ControlType.LINEAR_CONTINUOUS]
)
@pytest.mark.parametrize(
    "integration_rule",
    [QuadratureRule.RECTANGLE_LEFT, QuadratureRule.APPROXIMATE_TRAPEZOIDAL, QuadratureRule.TRAPEZOIDAL],
)
def test_pendulum(control_type, integration_rule, objective, phase_dynamics):
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_shooting=30,
        integration_rule=integration_rule,
        objective=objective,
        control_type=control_type,
        phase_dynamics=phase_dynamics,
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(5)
    sol = ocp.solve(solver)
    j_printed = sum_cost_function_output(sol)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    tau = controls["tau"]
    dt = sol.t_span()[0][-1]

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    if integration_rule == QuadratureRule.RECTANGLE_LEFT:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 36.077211633874164)
                npt.assert_almost_equal(j_printed, 36.077211633874164)
                npt.assert_almost_equal(tau[:, -1], np.array([-15.79894366, 0.0]))
            else:
                npt.assert_almost_equal(f[0, 0], 18.91863487850207)
                npt.assert_almost_equal(j_printed, 18.91863487850207)
                npt.assert_almost_equal(tau[:, -1], np.array([-17.24468626, 0.0]))

        elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            npt.assert_equal(np.isnan(tau[:, -1]), np.array([False, False]))
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 36.077211633874185)
                npt.assert_almost_equal(j_printed, 36.077211633874185)

                controls = sol.decision_controls(to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS])
                states = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.KEYS])
                out = 0
                for i, fcn in enumerate(ocp.nlp[0].J[0].weighted_function):
                    out += fcn(
                        0,
                        dt,
                        states[:, i : i + 2].reshape((-1, 1)),  # States
                        controls[:, i : i + 2].reshape((-1, 1)),  # Controls
                        [],  # Parameters
                        [],  # Algebraic states
                        [],  # numerical timeseries
                        ocp.nlp[0].J[0].weight,  # Weight
                        [],  # Target
                    )
                npt.assert_almost_equal(float(out[0, 0]), 36.077211633874185)
            else:
                npt.assert_almost_equal(f[0, 0], 18.918634878502065)
                npt.assert_almost_equal(j_printed, 18.918634878502065)

        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 52.0209218166193)
                npt.assert_almost_equal(j_printed, 52.0209218166193)
            else:
                npt.assert_almost_equal(f[0, 0], 18.844221574687065)
                npt.assert_almost_equal(j_printed, 18.844221574687065)

        else:
            raise NotImplementedError("Control type not implemented yet")

    elif integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                npt.assert_almost_equal(tau[:, -1], np.array([-15.79894366, 0.0]))
                npt.assert_almost_equal(f[0, 0], 36.077211633874164)
                npt.assert_almost_equal(j_printed, 36.077211633874164)
            else:
                npt.assert_almost_equal(tau[:, -1], np.array([-17.24468626, 0.0]))
                npt.assert_almost_equal(f[0, 0], 18.91863487850206)
                npt.assert_almost_equal(j_printed, 18.91863487850206)
        elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            npt.assert_equal(np.isnan(tau[:, -1]), np.array([False, False]))
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 36.077211633874164)
                npt.assert_almost_equal(j_printed, 36.077211633874164)
            else:
                npt.assert_almost_equal(f[0, 0], 18.91863487850206)
                npt.assert_almost_equal(j_printed, 18.91863487850206)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 52.0209218166202)
                npt.assert_almost_equal(j_printed, 52.0209218166202)
            else:
                npt.assert_almost_equal(f[0, 0], 18.844221574687094)
                npt.assert_almost_equal(j_printed, 18.844221574687094)
    elif integration_rule == QuadratureRule.TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                npt.assert_almost_equal(tau[:, -1], np.array([-15.79894366, 0.0]))
                npt.assert_almost_equal(f[0, 0], 36.077211633874164)
                npt.assert_almost_equal(j_printed, 36.077211633874164)
            else:
                npt.assert_almost_equal(tau[:, -1], np.array([-15.3519514, 0.0]))
                npt.assert_almost_equal(f[0, 0], 18.112963129413707)
                npt.assert_almost_equal(j_printed, 18.112963129413707)
        elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            npt.assert_equal(np.isnan(tau[:, -1]), np.array([0, 0]))
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 36.077211633874384)
                npt.assert_almost_equal(j_printed, 36.077211633874384)
            else:
                npt.assert_almost_equal(f[0, 0], 17.944878542423062)
                npt.assert_almost_equal(j_printed, 17.944878542423062)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 34.52084504124038)
                npt.assert_almost_equal(j_printed, 34.52084504124038)
            else:
                npt.assert_almost_equal(f[0, 0], 17.72098984554101)
                npt.assert_almost_equal(j_printed, 17.72098984554101)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("objective", ["torque", "qdot"])
@pytest.mark.parametrize("control_type", [ControlType.CONSTANT])
@pytest.mark.parametrize(
    "integration_rule",
    [
        QuadratureRule.RECTANGLE_LEFT,
        QuadratureRule.APPROXIMATE_TRAPEZOIDAL,
        QuadratureRule.TRAPEZOIDAL,
        QuadratureRule.RECTANGLE_RIGHT,
        QuadratureRule.MIDPOINT,
    ],
)
def test_pendulum_collocation(control_type, integration_rule, objective, phase_dynamics):
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    if integration_rule not in (
        QuadratureRule.RECTANGLE_LEFT,
        QuadratureRule.TRAPEZOIDAL,
        QuadratureRule.APPROXIMATE_TRAPEZOIDAL,
    ):
        with pytest.raises(
            NotImplementedError,
            match=f"{integration_rule} has not been implemented yet for objective functions.",
        ):
            prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
                n_shooting=10,
                integration_rule=integration_rule,
                objective=objective,
                control_type=control_type,
                ode_solver=OdeSolver.COLLOCATION(),
                phase_dynamics=phase_dynamics,
            )
        return

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_shooting=10,
        integration_rule=integration_rule,
        objective=objective,
        control_type=control_type,
        ode_solver=OdeSolver.COLLOCATION(),
        phase_dynamics=phase_dynamics,
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(5)
    sol = ocp.solve(solver)
    j_printed = sum_cost_function_output(sol)

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    if integration_rule == QuadratureRule.RECTANGLE_LEFT:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 11.795040652982767)
                npt.assert_almost_equal(j_printed, 11.795040652982767)
            else:
                npt.assert_almost_equal(f[0, 0], 12.336208562756555)
                npt.assert_almost_equal(j_printed, 12.336208562756553)
    elif integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 11.795040652982767)
                npt.assert_almost_equal(j_printed, 11.795040652982767)
            else:
                npt.assert_almost_equal(f[0, 0], 12.336208562756559)
                npt.assert_almost_equal(j_printed, 12.336208562756559)
    elif integration_rule == QuadratureRule.TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 11.795040652982767)
                npt.assert_almost_equal(j_printed, 11.795040652982767)
            else:
                npt.assert_almost_equal(f[0, 0], 12.336208562756564)
                npt.assert_almost_equal(j_printed, 12.336208562756564)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("objective", ["torque", "qdot"])
@pytest.mark.parametrize(
    "control_type", [ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE, ControlType.LINEAR_CONTINUOUS]
)
@pytest.mark.parametrize(
    "integration_rule",
    [QuadratureRule.RECTANGLE_LEFT, QuadratureRule.APPROXIMATE_TRAPEZOIDAL, QuadratureRule.TRAPEZOIDAL],
)
def test_pendulum_target(control_type, integration_rule, objective, phase_dynamics):
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    if objective == "qdot":
        target = np.array(
            [
                [
                    0.0,
                    5.07107824,
                    5.29716282,
                    3.71599869,
                    2.69356547,
                    1.98544214,
                    1.43359131,
                    0.95250526,
                    0.48160389,
                    -0.0405487,
                    -0.70479343,
                    -1.70404096,
                    -3.67277459,
                    -12.07505107,
                    -12.48951239,
                    -7.04539771,
                    -4.69043379,
                    -3.210717,
                    -2.12953066,
                    -1.27300752,
                    -0.55915105,
                    0.0572217,
                    0.62354422,
                    1.18703801,
                    1.78233074,
                    2.45897744,
                    3.30436539,
                    4.51012445,
                    6.61008457,
                    9.64613529,
                ],
                [
                    0.0,
                    -5.09151461,
                    -5.289256,
                    -3.6730883,
                    -2.5999572,
                    -1.81858156,
                    -1.17533959,
                    -0.59215646,
                    -0.01743249,
                    0.59950305,
                    1.33570276,
                    2.36400437,
                    4.28266505,
                    12.4836082,
                    13.08487663,
                    8.28918906,
                    6.50446535,
                    5.52416911,
                    4.89409004,
                    4.45661472,
                    4.14198901,
                    3.91498066,
                    3.76221089,
                    3.68972717,
                    3.71430414,
                    3.87364337,
                    4.24824349,
                    5.02964499,
                    6.77303688,
                    9.65322827,
                ],
            ]
        )
    else:
        target = np.array(
            [
                [
                    6.01549798,
                    4.06186543,
                    2.1857845,
                    1.28263471,
                    0.74306671,
                    0.32536624,
                    -0.06946552,
                    -0.5067574,
                    -1.05437558,
                    -1.81359809,
                    -2.9872725,
                    -5.14327902,
                    -11.4338948,
                    -22.01970852,
                    -6.62382791,
                    -1.42881905,
                    1.03925748,
                    2.50114717,
                    3.41435707,
                    3.95852017,
                    4.21516739,
                    4.75104952,
                    5.61267393,
                    6.2718718,
                    6.67989085,
                    6.72242315,
                    6.11617043,
                    3.99273594,
                    -3.12543577,
                    -13.68877181,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )

    if integration_rule in (QuadratureRule.APPROXIMATE_TRAPEZOIDAL, QuadratureRule.TRAPEZOIDAL):
        target = np.concatenate((target, np.zeros((2, 1))), axis=1)

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_shooting=30,
        integration_rule=integration_rule,
        objective=objective,
        control_type=control_type,
        target=target,
        phase_dynamics=phase_dynamics,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(5)
    sol = ocp.solve(solver)
    j_printed = sum_cost_function_output(sol)

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    if integration_rule == QuadratureRule.RECTANGLE_LEFT:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 47.409664872029175)
                npt.assert_almost_equal(j_printed, 47.409664872029175)
            else:
                npt.assert_almost_equal(f[0, 0], 69.3969483839429, decimal=4)  # for windows
                npt.assert_almost_equal(j_printed, 69.3969483839429, decimal=4)
        elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 47.409664872029175)
                npt.assert_almost_equal(j_printed, 47.409664872029175)
            else:
                npt.assert_almost_equal(f[0, 0], 69.39694769390407, decimal=4)  # for windows
                npt.assert_almost_equal(j_printed, 69.39694769390407, decimal=4)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 47.18288247657242)
                npt.assert_almost_equal(j_printed, 47.18288247657242)
            else:
                npt.assert_almost_equal(f[0, 0], 49.52081908930845)
                npt.assert_almost_equal(j_printed, 49.52081908930845)
    elif integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 47.20218950613529)
                npt.assert_almost_equal(j_printed, 47.20218950613529)
            else:
                npt.assert_almost_equal(f[0, 0], 79.20445223944195, decimal=5)
                npt.assert_almost_equal(j_printed, 79.20445223944195, decimal=5)
        elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 47.20218950610398)
                npt.assert_almost_equal(j_printed, 47.20218950610398)
            else:
                npt.assert_almost_equal(f[0, 0], 79.20445223932471, decimal=5)
                npt.assert_almost_equal(j_printed, 79.20445223932471, decimal=5)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 48.49207210991624)
                npt.assert_almost_equal(j_printed, 48.49207210991624)
            else:
                npt.assert_almost_equal(f[0, 0], 47.038431660223246)
                npt.assert_almost_equal(j_printed, 47.038431660223246)
    elif integration_rule == QuadratureRule.TRAPEZOIDAL:
        if control_type == ControlType.CONSTANT:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 47.20218950613529)
                npt.assert_almost_equal(j_printed, 47.20218950613529)
            else:
                npt.assert_almost_equal(f[0, 0], 33.46130228108698)
                npt.assert_almost_equal(j_printed, 33.46130228108698)
        if control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 47.20218950610398)
                npt.assert_almost_equal(j_printed, 47.20218950610398)
            else:
                npt.assert_almost_equal(f[0, 0], 33.46130228109848)
                npt.assert_almost_equal(j_printed, 33.46130228109848)
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            if objective == "torque":
                npt.assert_almost_equal(f[0, 0], 43.72737954458251)
                npt.assert_almost_equal(j_printed, 43.72737954458251)
            else:
                npt.assert_almost_equal(f[0, 0], 35.61604299585224)
                npt.assert_almost_equal(j_printed, 35.61604299585224)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize(
    "integration_rule",
    [
        QuadratureRule.RECTANGLE_LEFT,
        QuadratureRule.APPROXIMATE_TRAPEZOIDAL,
        QuadratureRule.TRAPEZOIDAL,
    ],
)
def test_error_mayer_trapz(integration_rule, phase_dynamics):
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    with pytest.raises(
        ValueError,
        match="Mayer objective functions cannot be integrated, "
        "remove the argument integration_rule or use a Lagrange objective function",
    ):
        ocp = prepare_ocp(
            biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
            n_shooting=30,
            integration_rule=integration_rule,
            objective="mayer",
            control_type=ControlType.LINEAR_CONTINUOUS,
            phase_dynamics=phase_dynamics,
        )
