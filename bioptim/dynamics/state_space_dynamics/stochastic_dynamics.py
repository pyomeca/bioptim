from ..configure_variables import Controls, AlgebraicStates
from ..dynamics_functions import DynamicsFunctions
from ..dynamics_evaluation import DynamicsEvaluation
from ..ode_solvers import OdeSolver
from ...misc.enums import DefectType, ContactType
from ...optimization.problem_type import SocpType
from .torque_dynamics import TorqueDynamics


class StochasticTorqueDynamics(TorqueDynamics):
    """
    This class is used to create a model actuated through joint torques with stochastic variables.

    x = [q, qdot, ]
    u = [tau, stochastic_variables]
    a = [stochastic_variables]
    """

    def __init__(self, problem_type, with_cholesky, n_noised_tau, n_noise, n_noised_states, n_references):
        super().__init__(fatigue=None)
        self.control_configuration += [
            lambda ocp, nlp, as_states, as_controls, as_algebraic_states: Controls.K(
                ocp,
                nlp,
                as_states,
                as_controls,
                as_algebraic_states,
                n_noised_controls=n_noised_tau,
                n_references=n_references,
            ),
            lambda ocp, nlp, as_states, as_controls, as_algebraic_states: Controls.REF(
                ocp, nlp, as_states, as_controls, as_algebraic_states, n_references=n_references
            ),
        ]
        self.algebraic_configuration = [
            lambda ocp, nlp, as_states, as_controls, as_algebraic_states: AlgebraicStates.M(
                ocp, nlp, as_states, as_controls, as_algebraic_states, n_noised_states=n_noised_states
            ),
        ]

        if isinstance(problem_type, SocpType.TRAPEZOIDAL_EXPLICIT):
            raise RuntimeError(
                "The problem type TRAPEZOIDAL_EXPLICIT is not included in bioptim anymore, please use a custom configure."
            )
        if with_cholesky:
            self.control_configuration += [
                lambda ocp, nlp, as_states, as_controls, as_algebraic_states: Controls.CHOLESKY_COV(
                    ocp, nlp, as_states, as_controls, as_algebraic_states, n_noised_states=n_noised_states
                )
            ]
        else:
            self.control_configuration += [
                lambda ocp, nlp, as_states, as_controls, as_algebraic_states: Controls.COV(
                    ocp, nlp, as_states, as_controls, as_algebraic_states, n_noised_states=n_noised_states
                )
            ]

        if isinstance(problem_type, SocpType.TRAPEZOIDAL_IMPLICIT):
            self.control_configuration += [
                lambda ocp, nlp, as_states, as_controls, as_algebraic_states: Controls.A(
                    ocp, nlp, as_states, as_controls, as_algebraic_states, n_noised_states=n_noised_states
                ),
                lambda ocp, nlp, as_states, as_controls, as_algebraic_states: Controls.C(
                    ocp,
                    nlp,
                    as_states,
                    as_controls,
                    as_algebraic_states,
                    n_noised_states=n_noised_states,
                    n_noise=n_noise,
                ),
            ]

    def extra_dynamics(
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

        # Get states indices
        q_indices, qdot_indices = self.get_q_qdot_indices(nlp)

        # Get variables
        q, qdot, tau, external_forces = self.get_basic_variables(
            nlp, states, controls, parameters, algebraic_states, numerical_timeseries
        )

        if external_forces.shape != (0, 1):
            raise NotImplementedError("External forces are not implemented yet with stochastic torque driven dynamics.")
        tau += tau_feedback  # This is done here to avoid friction in the feedback torque function

        # Initialize dxdt
        dxdt = nlp.cx(nlp.states.shape, 1)
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.compute_qddot(nlp, q, qdot, tau, external_forces)
        dxdt[q_indices, 0] = dq
        dxdt[qdot_indices, 0] = ddq

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):

            if nlp.dynamics_type.ode_solver.defects_type != DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for stochastic torque driven dynamics."
                )

            DynamicsFunctions.no_states_mapping(nlp)
            slope_q, slope_qdot = self.get_basic_slopes(nlp)

            defects = nlp.cx(nlp.states.shape, 1)
            defects[q_indices, 0] = slope_q * nlp.dt - dq * nlp.dt
            defects[qdot_indices, 0] = slope_qdot * nlp.dt - ddq * nlp.dt

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
        raise NotImplementedError("Stochastic torque dynamics does not support rigid contact forces yet. ")
