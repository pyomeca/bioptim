from casadi import vertcat

from ..configure_variables import States, Controls
from ..dynamics_functions import DynamicsFunctions
from ..dynamics_evaluation import DynamicsEvaluation
from ..fatigue.fatigue_dynamics import FatigueList
from ..ode_solvers import OdeSolver
from ...misc.enums import DefectType
from .torque_dynamics import TorqueDynamics


class TorqueDerivativeDynamics(TorqueDynamics):
    def __init__(self, fatigue: FatigueList):
        super().__init__(fatigue=fatigue)
        self.state_configuration += [States.TAU]
        self.control_configuration = [Controls.TAUDOT]

    def get_basic_variables(self, nlp, states, controls, parameters, algebraic_states, numerical_timeseries):

        # Get variables from the right place
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.states["tau"], states)

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
        taudot = DynamicsFunctions.get(nlp.controls["taudot"], controls)

        # Initialize dxdt
        dxdt = nlp.cx(nlp.states.shape, 1)
        dxdt[q_indices, 0] = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        dxdt[qdot_indices, 0] = DynamicsFunctions.compute_qddot(nlp, q, qdot, tau, external_forces)
        dxdt[nlp.states["tau"].index, 0] = taudot

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):

            DynamicsFunctions.no_states_mapping(nlp)
            slope_q, slope_qdot = self.get_basic_slopes(nlp)
            slope_tau = nlp.states_dot["tau"].cx

            # Initialize defects
            defects = nlp.cx(nlp.states.shape, 1)

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:

                dxdt_defects = nlp.cx(nlp.states.shape, 1)
                dxdt_defects[q_indices, 0] = DynamicsFunctions.compute_qdot(nlp, q, qdot)
                dxdt_defects[qdot_indices, 0] = DynamicsFunctions.forward_dynamics(
                    nlp, q, qdot, tau, nlp.model.contact_types, external_forces
                )
                dxdt_defects[nlp.states["tau"].index, 0] = taudot

                slopes = nlp.cx(nlp.states.shape, 1)
                slopes[q_indices, 0] = slope_q
                slopes[qdot_indices, 0] = slope_qdot
                slopes[nlp.states["tau"].index, 0] = slope_tau

                defects = slopes * nlp.dt - dxdt_defects * nlp.dt

            elif nlp.dynamics_type.ode_solver.defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:

                defects[q_indices, 0] = slope_q * nlp.dt - qdot * nlp.dt

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
                defects[nlp.states["tau"].index, 0] = slope_tau * nlp.dt - taudot * nlp.dt
            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for torque driven dynamics."
                )

            defects = vertcat(defects, DynamicsFunctions.get_contact_defects(nlp, q, qdot, slope_qdot))

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)
