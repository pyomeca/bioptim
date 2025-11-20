from abc import abstractmethod

from casadi import vertcat

from ..configure_variables import States, Controls
from ..dynamics_functions import DynamicsFunctions
from ..dynamics_evaluation import DynamicsEvaluation
from ..fatigue.fatigue_dynamics import FatigueList
from ..ode_solvers import OdeSolver
from ...misc.enums import DefectType
from .abstract_dynamics import StateDynamicsWithContacts


class TorqueDynamics(StateDynamicsWithContacts):
    """
    This class is used to create a model actuated through joint torques.

    x = [q, qdot]
    u = [tau]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def state_configuration_functions(self):
        return [States.Q, States.QDOT]

    @property
    def control_configuration_functions(self):
        return [Controls.TAU]

    @property
    def algebraic_configuration_functions(self):
        return []

    @property
    def extra_configuration_functions(self):
        return []

    @staticmethod
    def get_q_qdot_indices(nlp):
        """
        Get the indices of the states and controls in the normal dynamics
        """
        return nlp.states["q"].index, nlp.states["qdot"].index

    def get_basic_slopes(self, nlp):
        """
        Get the slopes of the states in the normal dynamics.
        Please note that, we do not use DynamicsFunctions.get to get the slopes because we do not want them mapped
        """
        slope_q = nlp.states_dot["q"].cx
        slope_qdot = nlp.states_dot["qdot"].cx
        return slope_q, slope_qdot

    def get_basic_variables(self, nlp, states, controls, parameters, algebraic_states, numerical_timeseries):

        # Get variables from the right place
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get_fatigable_tau(nlp, states, controls)

        # Add additional torques
        tau += DynamicsFunctions.collect_tau(nlp, q, qdot, parameters)

        # Get external forces
        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return q, qdot, tau, external_forces

    def dynamics(
        self,
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ):

        # Get states indices
        q_indices, qdot_indices = self.get_q_qdot_indices(nlp)

        # Get variables
        q, qdot, tau, external_forces = self.get_basic_variables(
            nlp, states, controls, parameters, algebraic_states, numerical_timeseries
        )

        # Initialize dxdt
        dxdt = nlp.cx(nlp.states.shape, 1)
        dxdt[q_indices, 0] = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        dxdt[qdot_indices, 0] = DynamicsFunctions.compute_qddot(nlp, q, qdot, tau, external_forces)

        if nlp.model.fatigue is not None and "tau" in nlp.model.fatigue:
            dxdt = nlp.model.fatigue["tau"].dynamics(dxdt, nlp, states, controls)

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):

            DynamicsFunctions.no_states_mapping(nlp)
            slope_q, slope_qdot = self.get_basic_slopes(nlp)

            # Initialize defects
            defects = nlp.cx(nlp.states.shape, 1)

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:

                dxdt_defects = nlp.cx(nlp.states.shape, 1)
                dxdt_defects[q_indices, 0] = DynamicsFunctions.compute_qdot(nlp, q, qdot)
                dxdt_defects[qdot_indices, 0] = DynamicsFunctions.forward_dynamics(
                    nlp, q, qdot, tau, nlp.model.contact_types, external_forces
                )

                slopes = nlp.cx(nlp.states.shape, 1)
                slopes[q_indices, 0] = slope_q
                slopes[qdot_indices, 0] = slope_qdot

                # Get fatigue defects
                dxdt_defects, slopes = DynamicsFunctions.get_fatigue_defects(
                    "tau",
                    dxdt_defects,
                    slopes,
                    nlp,
                    states,
                    controls,
                )

                defects = slopes - dxdt_defects

            elif nlp.dynamics_type.ode_solver.defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
                if nlp.model.fatigue is not None:
                    raise NotImplementedError("Fatigue is not implemented yet with inverse dynamics defects.")

                defects[q_indices, 0] = slope_q - qdot

                tau_id = DynamicsFunctions.inverse_dynamics(
                    nlp,
                    q=q,
                    qdot=qdot,
                    qddot=slope_qdot,
                    contact_types=nlp.model.contact_types,
                    external_forces=external_forces,
                )
                tau_defects = tau - tau_id
                defects[qdot_indices, 0] = tau_defects
            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for torque driven dynamics."
                )

            defects = vertcat(defects, DynamicsFunctions.get_contact_defects(nlp, q, qdot, slope_qdot))

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    def get_rigid_contact_forces(
        self,
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ):
        q, qdot, tau, external_forces = self.get_basic_variables(
            nlp, states, controls, parameters, algebraic_states, numerical_timeseries
        )
        return nlp.model.rigid_contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)

    @property
    def extra_dynamics(self):
        return None
