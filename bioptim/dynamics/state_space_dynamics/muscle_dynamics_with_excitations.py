from casadi import vertcat

from ..configure_variables import States, Controls
from ..dynamics_functions import DynamicsFunctions
from ..dynamics_evaluation import DynamicsEvaluation
from ..fatigue.fatigue_dynamics import FatigueList
from ..ode_solvers import OdeSolver
from ...misc.enums import DefectType, ContactType
from ...misc.parameters_types import Bool
from .torque_dynamics import TorqueDynamics


class MusclesDynamicsWithExcitations(TorqueDynamics):
    """
    This class is used to create a model actuated through muscle activation.

    x = [q, qdot, muscles (if with_excitation)]
    u = [muscles, tau (if with_residual_torque)]
    """

    def __init__(self, with_residual_torque: Bool, fatigue: FatigueList = None):
        super().__init__(fatigue)

        self.state_configuration = [States.Q, States.QDOT]
        self.state_configuration += [States.MUSCLE_ACTIVATION]

        self.control_configuration = [Controls.MUSCLE_EXCITATION]
        if with_residual_torque:
            self.control_configuration = [Controls.TAU, Controls.MUSCLE_EXCITATION]

        self.algebraic_configuration = []
        self.functions = []
        self.with_residual_torque = with_residual_torque

    def get_basic_variables(
        self,
        nlp,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
    ):

        # Get variables from the right place
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        mus_activations = DynamicsFunctions.get(nlp.states["muscles"], states)
        fatigue_states, mus_activations = DynamicsFunctions.get_fatigue_states(states, nlp, mus_activations)

        # Compute the torques due to muscles
        muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations, fatigue_states)

        # Add additional torques
        if self.with_residual_torque:
            muscles_tau += DynamicsFunctions.get_fatigable_tau(nlp, states, controls)
        muscles_tau += DynamicsFunctions.collect_tau(nlp, q, qdot, parameters)

        # Get external forces
        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return q, qdot, muscles_tau, external_forces, mus_activations

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
        q, qdot, tau, external_forces, mus_activations = self.get_basic_variables(
            nlp, states, controls, parameters, algebraic_states, numerical_timeseries
        )

        # Initialize dxdt
        dxdt = nlp.cx(nlp.states.shape, 1)
        dxdt[q_indices, 0] = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        dxdt[qdot_indices, 0] = DynamicsFunctions.compute_qddot(nlp, q, qdot, tau, external_forces)

        mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
        dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations, mus_activations)
        dxdt[nlp.states["muscles"].index, 0] = dmus

        if nlp.model.fatigue is not None and "muscles" in nlp.model.fatigue:
            dxdt = nlp.model.fatigue["muscles"].dynamics(dxdt, nlp, states, controls)

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):

            DynamicsFunctions.no_states_mapping(nlp)
            slope_q, slope_qdot = self.get_basic_slopes(nlp)

            # Initialize defects
            defects = nlp.cx(nlp.states.shape, 1)

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:

                dxdt_defects = nlp.cx(nlp.states.shape, 1)
                ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, nlp.model.contact_types, external_forces)
                dxdt_defects[nlp.states["q"].index, 0] = DynamicsFunctions.compute_qdot(nlp, q, qdot)
                dxdt_defects[nlp.states["qdot"].index, 0] = ddq

                slopes = nlp.cx(nlp.states.shape, 1)
                slopes[nlp.states["q"].index, 0] = slope_q
                slopes[nlp.states["qdot"].index, 0] = slope_qdot

                mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
                dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations, mus_activations)
                dxdt_defects[nlp.states["muscles"].index, 0] = dmus
                slope_mus = nlp.states_dot["muscles"].cx
                slopes[nlp.states["muscles"].index, 0] = slope_mus

                if nlp.model.fatigue is not None and "muscles" in nlp.model.fatigue:
                    dxdt_defects = nlp.model.fatigue["muscles"].dynamics(dxdt_defects, nlp, states, controls)
                    state_keys = nlp.states.keys()
                    if state_keys[0] != "q" or state_keys[1] != "qdot":
                        raise NotImplementedError(
                            "The accession of muscles fatigue states is not implemented generically yet."
                        )

                    slopes_fatigue = nlp.cx()
                    fatigue_indices = []
                    for key in state_keys[2:]:
                        if not key.startswith("muscles_"):
                            raise NotImplementedError(
                                "The accession of muscles fatigue states is not implemented generically yet."
                            )
                        slopes_fatigue = vertcat(slopes_fatigue, nlp.states_dot[key].cx)
                        fatigue_indices += list(nlp.states[key].index)

                    slopes[fatigue_indices, 0] = slopes_fatigue

                defects = slopes - dxdt_defects

            elif nlp.dynamics_type.ode_solver.defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
                if nlp.model.fatigue is not None:
                    raise NotImplementedError("Fatigue is not implemented yet with inverse dynamics defects.")
                if ContactType.RIGID_EXPLICIT in nlp.model.contact_types:
                    raise NotImplementedError("Inverse dynamics, cannot be used with ContactType.RIGID_EXPLICIT yet")

                dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
                defects[nlp.states["q"].index, 0] = slope_q - dq

                external_forces = DynamicsFunctions.get_external_forces_from_contacts(
                    nlp, q, qdot, nlp.model.contact_types, external_forces
                )
                # TODO: We do not use DynamicsFunctions.inverse_dynamics here since tau is not in the variables (this should be refactored)
                tau_id = nlp.model.inverse_dynamics(with_contact=False)(
                    q, qdot, slope_qdot, external_forces, nlp.parameters.cx
                )
                tau_defects = tau - tau_id
                defects[nlp.states["qdot"].index, 0] = tau_defects

                mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
                dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations, mus_activations)
                slope_mus = nlp.states_dot["muscles"].cx
                defects[nlp.states["muscles"].index, 0] = slope_mus - dmus

            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for muscles driven dynamics."
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
        q, qdot, tau, external_forces, _ = self.get_basic_variables(
            nlp, states, controls, parameters, algebraic_states, numerical_timeseries
        )
        return nlp.model.rigid_contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)
