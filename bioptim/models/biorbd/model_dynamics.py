from typing import Callable
import biorbd_casadi as biorbd
import numpy as np

from bioptim.models.biorbd.external_forces import (
    ExternalForceSetTimeSeries,
    ExternalForceSetVariables,
)
from ...optimization.parameters import ParameterList
from .biorbd_model import BiorbdModel
from .multi_biorbd_model import MultiBiorbdModel
from .stochastic_biorbd_model import StochasticBiorbdModel
from .holonomic_biorbd_model import HolonomicBiorbdModel
from .variational_biorbd_model import VariationalBiorbdModel
from ..protocols.abstract_model_dynamics import (
    TorqueDynamics,
    StochasticTorqueDynamics,
    HolonomicTorqueDynamics,
    TorqueFreeFloatingBaseDynamics,
    StochasticTorqueFreeFloatingBaseDynamics,
    TorqueActivationDynamics,
    TorqueDerivativeDynamics,
    MusclesDynamics,
    JointAccelerationDynamics,
)
from ...misc.parameters_types import (
    Str,
    Int,
    Bool,
    NpArray,
    DM,
)
from ...misc.mapping import BiMappingList
from ...misc.enums import ContactType
from ...optimization.problem_type import SocpType


class TorqueBiorbdModel(BiorbdModel, TorqueDynamics):
    def __init__(
        self,
        bio_model: Str | biorbd.Model,
        friction_coefficients: np.ndarray = None,
        parameters: ParameterList = None,
        external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
        contact_types: list[ContactType] | tuple[ContactType] = (),
    ):
        BiorbdModel.__init__(self, bio_model, friction_coefficients, parameters, external_force_set, contact_types)
        TorqueDynamics.__init__(self)

    def serialize(self) -> tuple[Callable, dict]:
        return TorqueBiorbdModel, dict(bio_model=self.path,
                                       friction_coefficients=self.friction_coefficients,
                                       external_force_set=self.external_force_set,
                                       contact_types=self.contact_types)


class StochasticTorqueBiorbdModel(StochasticBiorbdModel, StochasticTorqueDynamics):
    def __init__(
        self,
        bio_model: Str | biorbd.Model,
        problem_type: SocpType,
        with_cholesky: Bool,
        n_references: Int,
        n_feedbacks: Int,
        n_noised_states: Int,
        n_noised_controls: Int,
        sensory_noise_magnitude: NpArray | DM,
        motor_noise_magnitude: NpArray | DM,
        sensory_reference: Callable,
        compute_torques_from_noise_and_feedback: Callable = None,
        motor_noise_mapping: BiMappingList = BiMappingList(),
        use_sx: Bool = False,
        parameters: ParameterList = None,
        friction_coefficients: NpArray = None,
    ):
        StochasticBiorbdModel.__init__(
            self,
            bio_model,
            problem_type,
            n_references,
            n_feedbacks,
            n_noised_states,
            n_noised_controls,
            sensory_noise_magnitude,
            motor_noise_magnitude,
            sensory_reference,
            compute_torques_from_noise_and_feedback,
            motor_noise_mapping,
            use_sx,
            parameters,
            friction_coefficients,
        )

        if "tau" in self.motor_noise_mapping:
            n_noised_tau = len(self.motor_noise_mapping["tau"].to_first.map_idx)
        else:
            n_noised_tau = self.nb_tau
        n_noise = self.motor_noise_magnitude.shape[0] + self.sensory_noise_magnitude.shape[0]
        n_noised_states = 2 * n_noised_tau

        StochasticTorqueDynamics.__init__(
            self, problem_type, with_cholesky, n_noised_tau, n_noise, n_noised_states, n_references
        )

    def serialize(self) -> tuple[Callable, dict]:
        return StochasticTorqueBiorbdModel, dict(bio_model=self.path,
                                       friction_coefficients=self.friction_coefficients)


class HolonomicTorqueBiorbdModel(HolonomicBiorbdModel, HolonomicTorqueDynamics):
    def __init__(
        self, bio_model: str | biorbd.Model, friction_coefficients: np.ndarray = None, parameters: ParameterList = None
    ):
        HolonomicBiorbdModel.__init__(self, bio_model, friction_coefficients, parameters)
        HolonomicTorqueDynamics.__init__(self)

    def serialize(self) -> tuple[Callable, dict]:
        return HolonomicTorqueBiorbdModel, dict(bio_model=self.path,
                                       friction_coefficients=self.friction_coefficients)


class VariationalTorqueBiorbdModel(VariationalBiorbdModel, HolonomicTorqueDynamics):
    def __init__(
        self, bio_model: str | biorbd.Model, friction_coefficients: np.ndarray = None, parameters: ParameterList = None
    ):
        VariationalBiorbdModel.__init__(self, bio_model, friction_coefficients, parameters)
        HolonomicTorqueDynamics.__init__(self)

    def serialize(self) -> tuple[Callable, dict]:
        return VariationalTorqueBiorbdModel, dict(bio_model=self.path,
                                       friction_coefficients=self.friction_coefficients)


class TorqueFreeFloatingBaseBiorbdModel(BiorbdModel, TorqueFreeFloatingBaseDynamics):
    def __init__(
        self,
        bio_model: str | biorbd.Model,
        friction_coefficients: np.ndarray = None,
        parameters: ParameterList = None,
        external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
        contact_types: list[ContactType] | tuple[ContactType] = (),
    ):
        BiorbdModel.__init__(self, bio_model, friction_coefficients, parameters, external_force_set, contact_types)
        TorqueFreeFloatingBaseDynamics.__init__(self)

    def serialize(self) -> tuple[Callable, dict]:
        return TorqueFreeFloatingBaseBiorbdModel, dict(bio_model=self.path,
                                       friction_coefficients=self.friction_coefficients,
                                       external_force_set=self.external_force_set,
                                       contact_types=self.contact_types)


class StochasticTorqueFreeFloatingBaseBiorbdModel(StochasticBiorbdModel, StochasticTorqueFreeFloatingBaseDynamics):
    def __init__(
        self,
        bio_model: Str | biorbd.Model,
        problem_type: SocpType,
        with_cholesky: Bool,
        n_references: Int,
        n_feedbacks: Int,
        n_noised_states: Int,
        n_noised_controls: Int,
        sensory_noise_magnitude: NpArray | DM,
        motor_noise_magnitude: NpArray | DM,
        sensory_reference: Callable,
        compute_torques_from_noise_and_feedback: Callable = None,
        motor_noise_mapping: BiMappingList = BiMappingList(),
        use_sx: Bool = False,
        parameters: ParameterList = None,
        friction_coefficients: NpArray = None,
    ):
        StochasticBiorbdModel.__init__(
            self,
            bio_model,
            problem_type,
            n_references,
            n_feedbacks,
            n_noised_states,
            n_noised_controls,
            sensory_noise_magnitude,
            motor_noise_magnitude,
            sensory_reference,
            compute_torques_from_noise_and_feedback,
            motor_noise_mapping,
            use_sx,
            parameters,
            friction_coefficients,
        )

        if "tau_joints" in self.motor_noise_mapping:
            n_noised_tau = len(self.motor_noise_mapping["tau_joints"].to_first.map_idx)
        else:
            n_noised_tau = self.nb_tau
        n_noise = self.motor_noise_magnitude.shape[0] + self.sensory_noise_magnitude.shape[0]
        n_noised_states = 2 * n_noised_tau

        StochasticTorqueFreeFloatingBaseDynamics.__init__(
            self, problem_type, with_cholesky, n_noised_tau, n_noise, n_noised_states, n_references
        )

    def serialize(self) -> tuple[Callable, dict]:
        return StochasticTorqueFreeFloatingBaseBiorbdModel, dict(bio_model=self.path,
                                       friction_coefficients=self.friction_coefficients)

class TorqueActivationBiorbdModel(BiorbdModel, TorqueActivationDynamics):
    def __init__(
        self,
        bio_model: str | biorbd.Model,
        with_residual_torque: Bool = False,
        friction_coefficients: np.ndarray = None,
        parameters: ParameterList = None,
        external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
        contact_types: list[ContactType] | tuple[ContactType] = (),
    ):
        BiorbdModel.__init__(self, bio_model, friction_coefficients, parameters, external_force_set, contact_types)
        TorqueActivationDynamics.__init__(self, with_residual_torque)

    def serialize(self) -> tuple[Callable, dict]:
        return TorqueActivationBiorbdModel, dict(bio_model=self.path,
                                       friction_coefficients=self.friction_coefficients,
                                       external_force_set=self.external_force_set,
                                       contact_types=self.contact_types)

class TorqueDerivativeBiorbdModel(BiorbdModel, TorqueDerivativeDynamics):
    def __init__(
        self,
        bio_model: str | biorbd.Model,
        friction_coefficients: np.ndarray = None,
        parameters: ParameterList = None,
        external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
        contact_types: list[ContactType] | tuple[ContactType] = (),
    ):
        BiorbdModel.__init__(self, bio_model, friction_coefficients, parameters, external_force_set, contact_types)
        TorqueDerivativeDynamics.__init__(self)

    def serialize(self) -> tuple[Callable, dict]:
        return TorqueDerivativeBiorbdModel, dict(bio_model=self.path,
                                       friction_coefficients=self.friction_coefficients,
                                       external_force_set=self.external_force_set,
                                       contact_types=self.contact_types)


class MusclesBiorbdModel(BiorbdModel, MusclesDynamics):
    def __init__(
        self,
        bio_model: Str | biorbd.Model,
        with_residual_torque: Bool = False,
        with_excitation: Bool = False,
        friction_coefficients: np.ndarray = None,
        parameters: ParameterList = None,
        external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
        contact_types: list[ContactType] | tuple[ContactType] = (),
    ):
        BiorbdModel.__init__(self, bio_model, friction_coefficients, parameters, external_force_set, contact_types)
        MusclesDynamics.__init__(self, with_residual_torque, with_excitation)

    def serialize(self) -> tuple[Callable, dict]:
        return MusclesBiorbdModel, dict(bio_model=self.path,
                                       friction_coefficients=self.friction_coefficients,
                                       external_force_set=self.external_force_set,
                                       contact_types=self.contact_types)

class JointAccelerationBiorbdModel(BiorbdModel, JointAccelerationDynamics):
    def __init__(
        self,
        bio_model: str | biorbd.Model,
        friction_coefficients: np.ndarray = None,
        parameters: ParameterList = None,
        external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
        contact_types: list[ContactType] | tuple[ContactType] = (),
    ):
        BiorbdModel.__init__(self, bio_model, friction_coefficients, parameters, external_force_set, contact_types)
        JointAccelerationDynamics.__init__(self)

    def serialize(self) -> tuple[Callable, dict]:
        return JointAccelerationBiorbdModel, dict(bio_model=self.path,
                                       friction_coefficients=self.friction_coefficients,
                                       external_force_set=self.external_force_set,
                                       contact_types=self.contact_types)


class MultiTorqueBiorbdModel(MultiBiorbdModel, TorqueDynamics):
    def __init__(
        self,
        bio_model: Str | biorbd.Model,
        extra_bio_models: tuple[str | biorbd.Model | BiorbdModel, ...] = (),
    ):
        MultiBiorbdModel.__init__(self, bio_model, extra_bio_models)
        TorqueDynamics.__init__(self)

    def serialize(self) -> tuple[Callable, dict]:
        return MultiTorqueBiorbdModel, dict(bio_model=self.path)


# TODO: add variational
