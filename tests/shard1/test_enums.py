from bioptim import (
    Axis,
    Node,
    CostType,
    VariableType,
    DefectType,
    InterpolationType,
    Shooting,
    MagnitudeType,
    MultiCyclicCycleSolutions,
    PlotType,
    ControlType,
    SolutionIntegrator,
    QuadratureRule,
    SoftContactDynamics,
    ExternalForceType,
    ReferenceFrame,
)

from bioptim.misc.enums import SolverType, PenaltyType, ConstraintType


def test_axis():
    assert Axis.X.value == 0
    assert Axis.Y.value == 1
    assert Axis.Z.value == 2

    # verify the number of elements
    assert len(Axis) == 3


def test_solver_type():
    assert SolverType.IPOPT.value == "Ipopt"
    assert SolverType.ACADOS.value == "ACADOS"
    assert SolverType.SQP.value == "SqpMethod"
    assert SolverType.NONE.value == None

    # verify the number of elements
    assert len(SolverType) == 4


def test_node():
    assert Node.START.value == "start"
    assert Node.MID.value == "mid"
    assert Node.INTERMEDIATES.value == "intermediates"
    assert Node.PENULTIMATE.value == "penultimate"
    assert Node.END.value == "end"
    assert Node.ALL.value == "all"
    assert Node.ALL_SHOOTING.value == "all_shooting"
    assert Node.TRANSITION.value == "transition"
    assert Node.MULTINODES.value == "multinodes"
    assert Node.DEFAULT.value == "default"

    # verify the number of elements
    assert len(Node) == 10


def test_interpolation_type():
    assert InterpolationType.CONSTANT.value == 0
    assert InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT.value == 1
    assert InterpolationType.LINEAR.value == 2
    assert InterpolationType.EACH_FRAME.value == 3
    assert InterpolationType.ALL_POINTS.value == 4
    assert InterpolationType.SPLINE.value == 5
    assert InterpolationType.CUSTOM.value == 6

    # verify the number of elements
    assert len(InterpolationType) == 7


def test_shooting():
    assert Shooting.MULTIPLE.value == "Multiple"
    assert Shooting.SINGLE.value == "Single"
    assert Shooting.SINGLE_DISCONTINUOUS_PHASE.value == "Single discontinuous phase"

    # verify the number of elements
    assert len(Shooting) == 3


def test_cost_type():
    assert CostType.OBJECTIVES.value == "Objectives"
    assert CostType.CONSTRAINTS.value == "Constraints"
    assert CostType.ALL.value == "All"

    # verify the number of elements
    assert len(CostType) == 3


def test_plot_type():
    assert PlotType.PLOT.value == 0
    assert PlotType.INTEGRATED.value == 1
    assert PlotType.STEP.value == 2
    assert PlotType.POINT.value == 3

    # verify the number of elements
    assert len(PlotType) == 4


def test_control_type():
    assert ControlType.CONSTANT.value == 1
    assert ControlType.LINEAR_CONTINUOUS.value == 2
    assert ControlType.CONSTANT_WITH_LAST_NODE.value == 3

    # verify the number of elements
    assert len(ControlType) == 4


def test_variable_type():
    assert VariableType.STATES.value == "states"
    assert VariableType.CONTROLS.value == "controls"
    assert VariableType.STATES_DOT.value == "states_dot"
    assert VariableType.ALGEBRAIC_STATES.value == "algebraic_states"

    # verify the number of elements
    assert len(VariableType) == 4


def test_solution_integrator():
    assert SolutionIntegrator.OCP.value == "OCP"
    assert SolutionIntegrator.SCIPY_RK23.value == "RK23"
    assert SolutionIntegrator.SCIPY_RK45.value == "RK45"
    assert SolutionIntegrator.SCIPY_DOP853.value == "DOP853"
    assert SolutionIntegrator.SCIPY_BDF.value == "BDF"
    assert SolutionIntegrator.SCIPY_LSODA.value == "LSODA"

    # verify the number of elements
    assert len(SolutionIntegrator) == 6


def test_penalty_type():
    assert PenaltyType.USER.value == "user"
    assert PenaltyType.INTERNAL.value == "internal"

    # verify the number of elements
    assert len(PenaltyType) == 2


def test_constraint_type():
    assert ConstraintType.IMPLICIT.value == "implicit"

    # verify the number of elements
    assert len(ConstraintType) == 1


def test_quadrature_rule():
    assert QuadratureRule.DEFAULT.value == "default"
    assert QuadratureRule.RECTANGLE_LEFT.value == "rectangle_left"
    assert QuadratureRule.RECTANGLE_RIGHT.value == "rectangle_right"
    assert QuadratureRule.MIDPOINT.value == "midpoint"
    assert QuadratureRule.APPROXIMATE_TRAPEZOIDAL.value == "approximate_trapezoidal"
    assert QuadratureRule.TRAPEZOIDAL.value == "trapezoidal"

    # verify the number of elements
    assert len(QuadratureRule) == 6


def test_soft_contact_dynamics():
    assert SoftContactDynamics.ODE.value == "ode"
    assert SoftContactDynamics.CONSTRAINT.value == "constraint"

    # verify the number of elements
    assert len(SoftContactDynamics) == 2


def test_defect_type():
    assert DefectType.EXPLICIT.value == "explicit"
    assert DefectType.IMPLICIT.value == "implicit"
    assert DefectType.NOT_APPLICABLE.value == "not_applicable"

    # verify the number of elements
    assert len(DefectType) == 3


def test_magnitude_type():
    assert MagnitudeType.ABSOLUTE.value == "absolute"
    assert MagnitudeType.RELATIVE.value == "relative"

    # verify the number of elements
    assert len(MagnitudeType) == 2


def test_multi_cyclic_cycle_solutions():
    assert MultiCyclicCycleSolutions.NONE.value == "none"
    assert MultiCyclicCycleSolutions.FIRST_CYCLES.value == "first_cycles"
    assert MultiCyclicCycleSolutions.ALL_CYCLES.value == "all_cycles"

    # verify the number of elements
    assert len(MultiCyclicCycleSolutions) == 3


def test_external_forces_type():
    assert ExternalForceType.FORCE.value == "force"
    assert ExternalForceType.TORQUE.value == "torque"
    assert ExternalForceType.TORQUE_AND_FORCE.value == "force_and_torque"

    # verify the number of elements
    assert len(ExternalForceType) == 3


def test_reference_frame():
    assert ReferenceFrame.GLOBAL.value == "global"
    assert ReferenceFrame.LOCAL.value == "local"

    # verify the number of elements
    assert len(ReferenceFrame) == 2
