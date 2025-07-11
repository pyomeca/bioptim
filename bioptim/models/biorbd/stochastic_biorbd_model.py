from typing import Callable

from casadi import DM

from .biorbd_model import BiorbdModel
from ...dynamics.dynamics_functions import DynamicsFunctions
from ...misc.mapping import BiMappingList
from ...optimization.parameters import ParameterList
from ...optimization.variable_scaling import VariableScaling
from ...optimization.problem_type import SocpType
from ..protocols.abstract_model_dynamics import DynamicalModel

from ...misc.parameters_types import Int, Str, Bool, NpArray


def _compute_torques_from_noise_and_feedback_default(
    nlp, time, states, controls, parameters, algebraic_states, numerical_timeseries, sensory_noise, motor_noise
):
    tau_nominal = DynamicsFunctions.get(nlp.controls["tau"], controls)

    ref = DynamicsFunctions.get(nlp.controls["ref"], controls)
    k = DynamicsFunctions.get(nlp.controls["k"], controls)
    from bioptim import StochasticBioModel

    k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)

    sensory_input = nlp.model.sensory_reference(
        time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp
    )
    tau_fb = k_matrix @ ((sensory_input - ref) + sensory_noise)

    tau_motor_noise = motor_noise

    tau = tau_nominal + tau_fb + tau_motor_noise

    return tau


class StochasticBiorbdModel(BiorbdModel):
    """
    This class allows to define a biorbd model.
    """

    def __init__(
        self,
        bio_model: list | tuple | DynamicalModel,
        problem_type: SocpType,
        n_references: Int,
        n_feedbacks: Int,
        n_noised_states: Int,
        n_noised_controls: Int,
        sensory_noise_magnitude: NpArray | DM,
        motor_noise_magnitude: NpArray | DM,
        sensory_reference: Callable,
        compute_torques_from_noise_and_feedback: Callable = _compute_torques_from_noise_and_feedback_default,
        motor_noise_mapping: BiMappingList = BiMappingList(),
        use_sx: Bool = False,
        parameters: ParameterList = None,
        friction_coefficients: NpArray = None,
        **kwargs,
    ):
        if parameters is None:
            parameters = ParameterList(use_sx=use_sx)
        parameters.add(
            "motor_noise",
            lambda model, param: None,
            size=motor_noise_magnitude.shape[0],
            scaling=VariableScaling("motor_noise", [1.0] * motor_noise_magnitude.shape[0]),
        )
        parameters.add(
            "sensory_noise",
            lambda model, param: None,
            size=sensory_noise_magnitude.shape[0],
            scaling=VariableScaling("sensory_noise", [1.0] * sensory_noise_magnitude.shape[0]),
        )
        super().__init__(
            bio_model=(bio_model if isinstance(bio_model, str) else bio_model.model),
            parameters=parameters,
            friction_coefficients=friction_coefficients,
            **kwargs,
        )
        self.problem_type = problem_type

        self.motor_noise_magnitude = motor_noise_magnitude
        self.sensory_noise_magnitude = sensory_noise_magnitude

        if compute_torques_from_noise_and_feedback is None:
            compute_torques_from_noise_and_feedback = _compute_torques_from_noise_and_feedback_default
        self.compute_torques_from_noise_and_feedback = compute_torques_from_noise_and_feedback

        self.sensory_reference = sensory_reference

        self.motor_noise_mapping = motor_noise_mapping

        self.n_references = n_references
        self.n_feedbacks = n_feedbacks
        self.n_noised_states = n_noised_states
        self.n_noise = motor_noise_magnitude.shape[0] + sensory_noise_magnitude.shape[0]
        self.n_noised_controls = n_noised_controls
        if motor_noise_mapping is not None and "tau" in motor_noise_mapping:
            if self.n_noised_controls != len(motor_noise_mapping["tau"].to_first.map_idx):
                raise RuntimeError("The number of noised controls must be equal to the number of tau mapping.")

        self.matrix_shape_k = (self.n_noised_controls, self.n_references)
        self.matrix_shape_c = (self.n_noised_states, self.n_noise)
        self.matrix_shape_a = (self.n_noised_states, self.n_noised_states)
        self.matrix_shape_cov = (self.n_noised_states, self.n_noised_states)
        self.matrix_shape_cov_cholesky = (self.n_noised_states, self.n_noised_states)
        self.matrix_shape_m = (self.n_noised_states, self.n_noised_states)
