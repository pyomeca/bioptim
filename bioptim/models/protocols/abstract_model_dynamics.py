
from casadi import vertcat, horzcat

from ...dynamics.configure_variables import States, Controls, AlgebraicStates
from ...dynamics.dynamics_functions import DynamicsFunctions, DynamicsEvaluation
from ...dynamics.fatigue.fatigue_dynamics import FatigueList
from ...dynamics.ode_solvers import OdeSolver
from ...misc.enums import DefectType, ContactType


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
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
            fatigue: FatigueList = None,
    ):

        # Get variables from the right place
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get_fatigable_tau(nlp, states, controls, fatigue)

        # Add additional torques
        tau += DynamicsFunctions.collect_tau(nlp, q, qdot, parameters, states, controls, fatigue)

        # Get external forces
        external_forces = nlp.get_external_forces("external_forces", states, controls, algebraic_states, numerical_timeseries)

        # Initialize dxdt
        dxdt = nlp.cx(nlp.states.shape, 1)
        dxdt[nlp.states["q"].index, 0] = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        dxdt[nlp.states["qdot"].index, 0] = DynamicsFunctions.compute_qddot(nlp, q, qdot, tau, external_forces)

        if fatigue is not None and "tau" in fatigue:
            dxdt = fatigue["tau"].dynamics(dxdt, nlp, states, controls)

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):

            # Initialize defects
            defects = nlp.cx(nlp.states.shape, 1)

            for key in nlp.states.keys():
                if nlp.variable_mappings[key].actually_does_a_mapping():
                    raise NotImplementedError(
                        f"COLLOCATION transcription is not compatible with mapping for states. "
                        "Please note that concept of states mapping in already sketchy on it's own, but is particularly not appropriate for COLLOCATION transcriptions."
                    )

            # Do not use DynamicsFunctions.get to get the slopes because we do not want them mapped
            slope_q = nlp.states_dot["q"].cx
            slope_qdot = nlp.states_dot["qdot"].cx


            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:

                dxdt_defects = nlp.cx(nlp.states.shape, 1)
                ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, nlp.model.contact_types, external_forces)
                dxdt_defects[nlp.states["qdot"].index, 0] = ddq

                slopes = nlp.cx(nlp.states.shape, 1)
                slopes[nlp.states["q"].index, 0] = slope_q
                slopes[nlp.states["qdot"].index, 0] = slope_qdot

                # Get fatigue defects
                slopes, dxdt_defects = DynamicsFunctions.get_fatigue_defects(
                    "tau", dxdt_defects, slopes, nlp, states, controls, fatigue,
                )

                defects = slopes * nlp.dt - dxdt_defects * nlp.dt

            elif nlp.dynamics_type.ode_solver.defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
                if fatigue is not None:
                    raise NotImplementedError("Fatigue is not implemented yet with inverse dynamics defects.")

                defects[nlp.states["q"].index, 0] = slope_q * nlp.dt - qdot * nlp.dt

                tau_id = DynamicsFunctions.inverse_dynamics(
                    nlp,
                    q=q,
                    qdot=qdot,
                    qddot=slope_qdot,
                    contact_types=nlp.model.contact_types,
                    external_forces=external_forces,
                )
                tau_defects = tau - tau_id
                defects[nlp.states["qdot"].index, 0] = tau_defects
            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for torque driven dynamics."
                )

            defects = vertcat(defects, DynamicsFunctions.get_contact_defects(nlp, q, qdot, slope_qdot))

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)