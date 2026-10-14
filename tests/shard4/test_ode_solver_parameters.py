"""
Tests of the parameters declared by the ode solvers.
"""

from bioptim import (
    DynamicsOptions,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    TorqueBiorbdModel,
)

from ..utils import TestUtils


class _ParametrizedRK4(OdeSolver.RK4):
    """An ode solver needing a parameter that does not modify the model."""

    def declare_parameters(self, parameters, model):
        parameters.add("solver_parameter", function=None, size=model.nb_q)


def _pendulum() -> TorqueBiorbdModel:
    return TorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod")


def test_default_ode_solver_declares_no_parameter():
    parameters = ParameterList(use_sx=False)

    OdeSolver.RK4().declare_parameters(parameters, _pendulum())

    assert parameters.keys() == []


def test_ocp_with_default_ode_solver_has_no_parameter():
    ocp = OptimalControlProgram(
        _pendulum(), n_shooting=5, phase_time=1, dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4())
    )

    assert ocp.nlp[0].parameters.keys() == []


def test_ode_solver_can_declare_a_parameter_without_model_function():
    model = _pendulum()
    parameters = ParameterList(use_sx=False)

    _ParametrizedRK4().declare_parameters(parameters, model)

    assert parameters.keys() == ["solver_parameter"]
    assert parameters["solver_parameter"].size == model.nb_q
    assert parameters["solver_parameter"].function is None
    # There is no model function to apply, so applying the parameter to the model does nothing
    parameters["solver_parameter"].apply_parameter(model)
