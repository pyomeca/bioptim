from typing import Callable

import biorbd_casadi as biorbd
import numpy as np
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
    GeneralizedTorque,
    GeneralizedAcceleration,
)
from casadi import SX, MX, vertcat, horzcat, norm_fro, Function, DM

from bioptim.models.biorbd.external_forces import (
    ExternalForceSetTimeSeries,
    ExternalForceSetVariables,
)
from ..utils import _var_mapping, bounds_from_ranges, cache_function
from ...limits.path_conditions import Bounds
from ...misc.mapping import BiMapping, BiMappingList
from ...misc.utils import check_version
from ...dynamics.configure_variables import States, Controls, AlgebraicStates
from ...dynamics.dynamics_functions import DynamicsFunctions, DynamicsEvaluation
from ...dynamics.fatigue.fatigue_dynamics import FatigueList
from ...optimization.parameters import ParameterList
from ...misc.enums import DefectType
from .biorbd_model import BiorbdModel


class VariableCollector:
    @staticmethod
    def collect_tau(model, q, qdot, parameters, with_passive_torque, with_ligament, with_friction, nlp, states, controls, fatigue):
        # TODO: modify the fatigue call so that we do not need states, and controls ?
        tau = DynamicsFunctions.get_fatigable_tau(nlp, states, controls, fatigue)
        tau = tau + model.passive_joint_torque()(q, qdot, parameters) if with_passive_torque else tau
        tau = tau + model.ligament_joint_torque()(q, qdot, parameters) if with_ligament else tau
        tau = tau - model.friction_coefficients @ qdot if with_friction else tau
        return tau

    def collect_qddot(self, q, qdot, tau):
        return self.model.torque(q, qdot, self.parameters.cx)

class TorqueDynamics:
    """
    This class is used to create a model actuated through joint torques.

    x = [q, qdot]
    u = [tau]
    """
    def __init__(self):
        self.state_type = [States.Q, States.QDOT]
        self.control_type = [Controls.TAU]
        self.algebraic_type = []

    def dynamics(
            self,
            model,
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
            fatigue: FatigueList,
    ):

        # Get variables
        parameters = nlp.parameters.cx
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

        # Dynamics related quantities
        with_passive_torque = False  # TODO: change
        with_ligament = False
        with_friction = False
        tau += VariableCollector.collect_tau(model, q, qdot, parameters, with_passive_torque, with_ligament, with_friction, nlp, states, controls, fatigue)

        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)

        # Derivatives
        contact_type = model.contact_type
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, contact_type, external_forces)
        dxdt = vertcat(dq, ddq)

        if fatigue is not None and "tau" in fatigue:
            dxdt = fatigue["tau"].dynamics(dxdt, nlp, states, controls)

        # Defects
        defects = None
        # TODO: contacts and fatigue to be handled with implicit dynamics
        if model.dynamics_type.ode_solver.defects_type == DefectType.IMPLICIT:
            if len(contact_type) == 0 and fatigue is None:
                qddot = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
                tau_id = DynamicsFunctions.inverse_dynamics(nlp, q, qdot, qddot, contact_type, external_forces)
                defects = nlp.cx(dq.shape[0] + tau_id.shape[0], tau_id.shape[1])

                dq_defects = []
                for _ in range(tau_id.shape[1]):
                    dq_defects.append(
                        dq
                        - DynamicsFunctions.compute_qdot(
                            nlp,
                            q,
                            DynamicsFunctions.get(nlp.states_dot.scaled["qdot"], nlp.states_dot.scaled.cx),
                        )
                    )
                defects[: dq.shape[0], :] = horzcat(*dq_defects)
                # We modified on purpose the size of the tau to keep the zero in the defects in order to respect the dynamics
                defects[dq.shape[0] :, :] = tau - tau_id

        return DynamicsEvaluation(dxdt, defects)



class TorqueBiorbdModel(BiorbdModel, TorqueDynamics):
    def __init__(
            self,
            bio_model: str | biorbd.Model,
            friction_coefficients: np.ndarray = None,
            parameters: ParameterList = None,
            external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None
    ):
        super(BiorbdModel).__init__(bio_model, friction_coefficients, parameters, external_force_set)
        super(TorqueDynamics).__init__()

