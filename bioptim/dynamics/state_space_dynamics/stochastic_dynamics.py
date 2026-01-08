from abc import abstractmethod

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

    def __init__(self, problem_type, with_cholesky, n_noised_tau, n_noise, **kwargs):
        if isinstance(problem_type, SocpType.TRAPEZOIDAL_EXPLICIT):
            raise RuntimeError(
                "The problem type TRAPEZOIDAL_EXPLICIT is not included in bioptim anymore, please use a custom configure."
            )

        super().__init__(fatigue=None, **kwargs)
        self.problem_type = problem_type
        self.with_cholesky = with_cholesky
        self.n_noised_tau = n_noised_tau
        self.n_noise = n_noise

    @property
    @abstractmethod
    def n_references(self):
        """
        The number of references for the feedback control.
        """

    @property
    @abstractmethod
    def n_noised_states(self):
        """
        The number of noised states.
        """

    @property
    def control_configuration_functions(self):
        val = super().control_configuration_functions + [
            lambda ocp, nlp: Controls.K(ocp, nlp, n_noised_controls=self.n_noised_tau, n_references=self.n_references),
            lambda ocp, nlp: Controls.REF(ocp, nlp, n_references=self.n_references),
        ]

        cov_func = Controls.CHOLESKY_COV if self.with_cholesky else Controls.COV
        val += [lambda ocp, nlp: cov_func(ocp, nlp, n_noised_states=self.n_noised_states)]

        if isinstance(self.problem_type, SocpType.TRAPEZOIDAL_IMPLICIT):
            val += [
                lambda ocp, nlp: Controls.A(ocp, nlp, n_noised_states=self.n_noised_states),
                lambda ocp, nlp: Controls.C(ocp, nlp, n_noised_states=self.n_noised_states, n_noise=self.n_noise),
            ]
        return val

    @property
    def algebraic_configuration_functions(self):
        return super().algebraic_configuration_functions + [
            lambda ocp, nlp: AlgebraicStates.M(ocp, nlp, n_noised_states=self.n_noised_states)
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
            defects[q_indices, 0] = slope_q - dq
            defects[qdot_indices, 0] = slope_qdot - ddq

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
