
from casadi import vertcat, horzcat

from ...dynamics.configure_variables import States, Controls, AlgebraicStates
from ...dynamics.dynamics_functions import DynamicsFunctions, DynamicsEvaluation
from ...dynamics.fatigue.fatigue_dynamics import FatigueList
from ...dynamics.ode_solvers import OdeSolver
from ...misc.enums import DefectType, ContactType
from ...optimization.problem_type import SocpType


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

    @staticmethod
    def get_basic_variables(nlp, states, controls, parameters, algebraic_states, numerical_timeseries, fatigue: FatigueList):

        # Get variables from the right place
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get_fatigable_tau(nlp, states, controls, fatigue)

        # Add additional torques
        tau += DynamicsFunctions.collect_tau(nlp, q, qdot, parameters, states, controls, fatigue)

        # Get external forces
        external_forces = nlp.get_external_forces("external_forces", states, controls, algebraic_states,
                                                  numerical_timeseries)
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
            fatigue: FatigueList = None,
    ):

        # Get variables
        q, qdot, tau, external_forces = self.get_basic_variables(nlp, states, controls, parameters, algebraic_states, numerical_timeseries, fatigue)

        # Initialize dxdt
        dxdt = nlp.cx(nlp.states.shape, 1)
        dxdt[nlp.states["q"].index, 0] = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        dxdt[nlp.states["qdot"].index, 0] = DynamicsFunctions.compute_qddot(nlp, q, qdot, tau, external_forces)

        if fatigue is not None and "tau" in fatigue:
            dxdt = fatigue["tau"].dynamics(dxdt, nlp, states, controls)

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):

            DynamicsFunctions.no_states_mapping(nlp)

            # Initialize defects
            defects = nlp.cx(nlp.states.shape, 1)

            # Do not use DynamicsFunctions.get to get the slopes because we do not want them mapped
            slope_q = nlp.states_dot["q"].cx
            slope_qdot = nlp.states_dot["qdot"].cx

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:

                dxdt_defects = nlp.cx(nlp.states.shape, 1)
                dxdt_defects[nlp.states["q"].index, 0] = DynamicsFunctions.compute_qdot(nlp, q, qdot)
                dxdt_defects[nlp.states["qdot"].index, 0] = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, nlp.model.contact_types, external_forces)

                slopes = nlp.cx(nlp.states.shape, 1)
                slopes[nlp.states["q"].index, 0] = slope_q
                slopes[nlp.states["qdot"].index, 0] = slope_qdot

                # Get fatigue defects
                dxdt_defects, slopes = DynamicsFunctions.get_fatigue_defects(
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

    def get_rigid_contact_forces(self, nlp, states, controls, parameters, algebraic_states, numerical_timeseries, fatigue: FatigueList = None):
        q, qdot, tau, external_forces = self.get_basic_variables(nlp, states, controls, parameters, algebraic_states,
                                                           numerical_timeseries, fatigue)
        return nlp.model.rigid_contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)



class StochasticTorqueDynamics(TorqueDynamics):
    """
    This class is used to create a model actuated through joint torques with stochastic variables.

    x = [q, qdot, stochastic_variables]
    u = [tau]
    """
    def __init__(self, problem_type, with_cholesky, n_noised_tau, n_noise, n_noised_states, n_references):
        super().__init__()
        self.control_type += [
            lambda ocp, nlp, as_states, as_controls, as_algebraic_states: Controls.K(ocp, nlp, n_noised_controls=n_noised_tau, n_references=n_references),
            lambda ocp, nlp, as_states, as_controls, as_algebraic_states: Controls.REF(ocp, nlp, n_references=n_references),
        ]
        self.algebraic_type = [
            lambda ocp, nlp, as_states, as_controls, as_algebraic_states: AlgebraicStates.M(ocp, nlp, n_noised_states=n_noised_states),
        ]

        if isinstance(problem_type, SocpType.TRAPEZOIDAL_EXPLICIT):
            raise RuntimeError(
                "The problem type TRAPEZOIDAL_EXPLICIT is not included in bioptim anymore, please use a custom configure."
            )
        if with_cholesky:
            self.control_type += [
                lambda ocp, nlp, as_states, as_controls, as_algebraic_states: Controls.CHOLESKY_COV(ocp, nlp, n_noised_states=n_noised_states)
            ]
        else:
            self.control_type += [
                lambda ocp, nlp, as_states, as_controls, as_algebraic_states: Controls.COV(ocp, nlp, n_noised_states=n_noised_states)
            ]

        if isinstance(problem_type, SocpType.TRAPEZOIDAL_IMPLICIT):
            self.control_type += [
                lambda ocp, nlp, as_states, as_controls, as_algebraic_states: Controls.A(ocp, nlp, n_noised_states=n_noised_states),
                lambda ocp, nlp, as_states, as_controls, as_algebraic_states: Controls.C(ocp, nlp, n_noised_states=n_noised_states, n_noise=n_noise)
            ]


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

        if (
            ContactType.SOFT_EXPLICIT in nlp.model.contact_types
            or ContactType.SOFT_IMPLICIT in nlp.model.contact_types
            or ContactType.RIGID_IMPLICIT in nlp.model.contact_types
        ):
            raise NotImplementedError(
                "soft contacts and implicit contacts not implemented yet with stochastic torque driven dynamics."
            )

        # Get stochastic variables
        motor_noise = DynamicsFunctions.get(nlp.parameters["motor_noise"], parameters)
        sensory_noise = DynamicsFunctions.get(nlp.parameters["sensory_noise"], parameters)
        tau_feedback = nlp.model.compute_torques_from_noise_and_feedback(
            nlp=nlp,
            time=time,
            states=states,
            controls=controls,
            parameters=parameters,
            algebraic_states=algebraic_states,
            numerical_timeseries=numerical_timeseries,
            sensory_noise=sensory_noise,
            motor_noise=motor_noise,
        )

        # Get variables
        q, qdot, tau, external_forces = self.get_basic_variables(
            nlp,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            fatigue=None)

        if external_forces.shape != (0, 1):
            raise NotImplementedError("External forces are not implemented yet with stochastic torque driven dynamics.")
        tau += tau_feedback  # This is done here to avoid friction in the feedback torque function

        # Initialize dxdt
        dxdt = nlp.cx(nlp.states.shape, 1)
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.compute_qddot(nlp, q, qdot, tau, external_forces)
        dxdt[nlp.states["q"].index, 0] = dq
        dxdt[nlp.states["qdot"].index, 0] = ddq

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):

            if nlp.dynamics_type.ode_solver.defects_type != DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for stochastic torque driven dynamics."
                )

            DynamicsFunctions.no_states_mapping(nlp)
            slope_q = nlp.states_dot["q"].cx
            slope_qdot = nlp.states_dot["qdot"].cx

            defects = nlp.cx(nlp.states.shape, 1)
            defects[nlp.states["q"].index, 0] = slope_q * nlp.dt - dq * nlp.dt
            defects[nlp.states["qdot"].index, 0] = slope_qdot * nlp.dt - ddq * nlp.dt

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    def get_rigid_contact_forces(self, nlp, states, controls, parameters, algebraic_states, numerical_timeseries, fatigue: FatigueList = None):
        raise NotImplementedError("Stochastic torque dynamics does not support rigid contact forces yet. ")


# class HolonomicTorqueDynamics:
#
# class TorqueFreeFloatingBaseDynamics:
#
# class StochasticBiorbdModel:
#
# class TorqueActivationDynamics:
#
# class TorqueDerivativeDynamics:
#
# class MusclesDynamics:
#
# class JointAccelerationDynamics:
