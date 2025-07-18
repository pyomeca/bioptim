from typing import TypeAlias
from casadi import vertcat, DM

from ...dynamics.configure_variables import States, Controls, AlgebraicStates
from ...dynamics.dynamics_functions import DynamicsFunctions
from ...dynamics.dynamics_evaluation import DynamicsEvaluation
from ...dynamics.fatigue.fatigue_dynamics import FatigueList
from ...dynamics.configure_variables import ConfigureVariables
from ...dynamics.ode_solvers import OdeSolver
from ...misc.enums import DefectType, ContactType
from ...misc.parameters_types import Bool, NpArray
from ...optimization.problem_type import SocpType
from ..protocols.abstract_model import AbstractModel


class TorqueDynamics(AbstractModel):
    """
    This class is used to create a model actuated through joint torques.

    x = [q, qdot]
    u = [tau]
    """

    def __init__(self, fatigue: FatigueList = None):
        super().__init__()
        self.state_configuration = [States.Q, States.QDOT]
        self.control_configuration = [Controls.TAU]
        self.algebraic_configuration = []
        self.functions = []
        self.fatigue = fatigue

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

                defects = slopes * nlp.dt - dxdt_defects * nlp.dt

            elif nlp.dynamics_type.ode_solver.defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
                if nlp.model.fatigue is not None:
                    raise NotImplementedError("Fatigue is not implemented yet with inverse dynamics defects.")

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


class HolonomicTorqueDynamics(AbstractModel):

    def __init__(self):
        super().__init__()
        self.state_configuration = [States.Q_U, States.QDOT_U]
        self.control_configuration = [Controls.TAU]
        self.algebraic_configuration = []
        self.functions = [
            lambda ocp, nlp: ConfigureVariables.configure_qv(ocp, nlp),
            lambda ocp, nlp: ConfigureVariables.configure_qdotv(ocp, nlp),
            lambda ocp, nlp: ConfigureVariables.configure_lagrange_multipliers_function(ocp, nlp),
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

        q_u = DynamicsFunctions.get(nlp.states["q_u"], states)
        qdot_u = DynamicsFunctions.get(nlp.states["qdot_u"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
        q_v_init = DM.zeros(nlp.model.nb_dependent_joints)

        qddot_u = nlp.model.partitioned_forward_dynamics()(q_u, qdot_u, q_v_init, tau)
        dxdt = vertcat(qdot_u, qddot_u)

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            slope_q = DynamicsFunctions.get(nlp.states_dot["qdot_u"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot_u"], nlp.states_dot.scaled.cx)
            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                qddot_u = nlp.model.partitioned_forward_dynamics()(q_u, qdot_u, q_v_init, tau)
                derivative = vertcat(qdot_u, qddot_u)
                defects = vertcat(slope_q, slope_qdot) * nlp.dt - derivative * nlp.dt
            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for holonomic torque driven dynamics."
                )

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    def get_rigid_contact_forces(self, time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp):
        return

    @property
    def extra_dynamics(self):
        return None


class VariationalTorqueDynamics(AbstractModel):

    def __init__(self):
        super().__init__()
        # If the model has no holonomic constraint, there will be no lambdas defined
        # TODO: QDOT are declared as parameters in VariationalOptimalControlProgram, these parameters should instead be
        # declared here as self.parameter_configuration = [Parameter.QDOT_INIT, Parameter.QDOT_END]
        self.state_configuration = [States.Q, Controls.LAMBDA]
        self.control_configuration = [Controls.TAU]
        self.algebraic_configuration = []
        self.functions = [
            lambda ocp, nlp: ConfigureVariables.configure_variational_functions(ocp, nlp),
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
        # For variational integrator, the traditional dynamics is skipped, but its declaration is mandatory in bioptim
        return DynamicsEvaluation(
            dxdt=nlp.cx(
                nlp.states.shape,
            ),
            defects=nlp.cx(0),
        )

    def get_rigid_contact_forces(self, time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp):
        return

    @property
    def extra_dynamics(self):
        return None


class TorqueFreeFloatingBaseDynamics(TorqueDynamics):
    """
    This class is used to create a model actuated through joint torques with a free floating base.

    x = [q_roots, q_joints, qdot_roots, qdot_joints]
    u = [tau_joints]
    """

    def __init__(self):
        super().__init__(fatigue=None)
        self.state_configuration = [States.Q_ROOTS, States.Q_JOINTS, States.QDOT_ROOTS, States.QDOT_JOINTS]
        self.control_configuration = [Controls.TAU_JOINTS]
        self.algebraic_configuration = []
        self.functions = []

    @staticmethod
    def get_q_qdot_indices(nlp):
        """
        Get the indices of the states and controls in the free floating base dynamics
        """
        return list(nlp.states["q_roots"].index) + list(nlp.states["q_joints"].index), list(
            nlp.states["qdot_roots"].index
        ) + list(nlp.states["qdot_joints"].index)

    def get_basic_variables(self, nlp, states, controls, parameters, algebraic_states, numerical_timeseries):

        if nlp.model.fatigue is not None:
            raise RuntimeError("Fatigue is not implemented yet for free floating base dynamics.")

        # Get variables from the right place
        q_roots = DynamicsFunctions.get(nlp.states["q_roots"], states)
        q_joints = DynamicsFunctions.get(nlp.states["q_joints"], states)
        qdot_roots = DynamicsFunctions.get(nlp.states["qdot_roots"], states)
        qdot_joints = DynamicsFunctions.get(nlp.states["qdot_joints"], states)
        tau_joints = DynamicsFunctions.get(nlp.controls["tau_joints"], controls)

        q_full = vertcat(q_roots, q_joints)
        qdot_full = vertcat(qdot_roots, qdot_joints)
        tau_full = vertcat(nlp.cx(nlp.model.nb_root, 1), tau_joints)

        # Add additional torques
        tau_full += DynamicsFunctions.collect_tau(nlp, q_full, qdot_full, parameters)

        # Get external forces
        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return q_full, qdot_full, tau_full, external_forces

    def get_basic_slopes(self, nlp):
        slope_q = vertcat(nlp.states_dot["q_roots"].cx, nlp.states_dot["q_joints"].cx)
        slope_qdot = vertcat(nlp.states_dot["qdot_roots"].cx, nlp.states_dot["qdot_joints"].cx)
        return slope_q, slope_qdot


class StochasticTorqueFreeFloatingBaseDynamics(TorqueFreeFloatingBaseDynamics, StochasticTorqueDynamics):
    """
    This class is used to create a model actuated through joint torques with a free floating base with stochastic variables.

    x = [q_roots, q_joints, qdot_roots, qdot_joints]
    u = [tau_joints, stochastic_variables]
    a = [stochastic_variables]
    """

    def __init__(self, problem_type, with_cholesky, n_noised_tau, n_noise, n_noised_states, n_references):
        super().__init__()
        StochasticTorqueDynamics.__init__(
            self, problem_type, with_cholesky, n_noised_tau, n_noise, n_noised_states, n_references
        )


class TorqueActivationDynamics(TorqueDynamics):
    def __init__(self, with_residual_torque: Bool, fatigue: FatigueList):
        super().__init__(fatigue=None)

        if with_residual_torque:
            self.control_configuration += [Controls.RESIDUAL_TAU]
        self.with_residual_torque = with_residual_torque
        self.fatigue = fatigue

    def get_basic_variables(self, nlp, states, controls, parameters, algebraic_states, numerical_timeseries):
        if nlp.model.fatigue is not None:
            raise NotImplementedError("Fatigue is not implemented yet for torque activation dynamics.")

        # Get variables from the right place
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau_activation = DynamicsFunctions.get(nlp.controls["tau"], controls)

        # Convert tau activations to joint torque
        tau = nlp.model.torque()(tau_activation, q, qdot, nlp.parameters.cx)
        if self.with_residual_torque:
            tau += DynamicsFunctions.get(nlp.controls["residual_tau"], controls)

        # Add additional torques
        tau += DynamicsFunctions.collect_tau(nlp, q, qdot, parameters)

        # Get external forces
        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return q, qdot, tau, external_forces


class TorqueDerivativeDynamics(TorqueDynamics):
    def __init__(self, fatigue: FatigueList):
        super().__init__(fatigue=None)
        self.state_configuration += [States.TAU]
        self.control_configuration = [Controls.TAUDOT]
        self.fatigue = fatigue

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


class MusclesDynamics(TorqueDynamics):
    """
    This class is used to create a model actuated through muscle activation.

    x = [q, qdot, muscles (if with_excitation)]
    u = [muscles, tau (if with_residual_torque)]
    """

    def __init__(self, with_residual_torque: Bool, with_excitation: Bool, fatigue: FatigueList = None):
        super().__init__(fatigue)

        self.state_configuration = [States.Q, States.QDOT]
        if with_excitation:
            self.state_configuration += [States.MUSCLE_ACTIVATION]

        if with_residual_torque:
            self.control_configuration = [Controls.TAU, Controls.MUSCLE_EXCITATION]
        else:
            self.control_configuration = [Controls.MUSCLE_EXCITATION]

        self.algebraic_configuration = []
        self.functions = []
        self.with_residual_torque = with_residual_torque
        self.with_excitation = with_excitation
        self.fatigue = fatigue

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
        if self.with_excitation:
            mus_activations = DynamicsFunctions.get(nlp.states["muscles"], states)
        else:
            mus_activations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
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

        has_excitation = True if "muscles" in nlp.states else False
        if has_excitation:
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

                has_excitation = True if "muscles" in nlp.states else False
                if has_excitation:
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

                defects = slopes * nlp.dt - dxdt_defects * nlp.dt

            elif nlp.dynamics_type.ode_solver.defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
                if nlp.model.fatigue is not None:
                    raise NotImplementedError("Fatigue is not implemented yet with inverse dynamics defects.")
                if ContactType.RIGID_EXPLICIT in nlp.model.contact_types:
                    raise NotImplementedError("Inverse dynamics, cannot be used with ContactType.RIGID_EXPLICIT yet")

                dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
                defects[nlp.states["q"].index, 0] = slope_q * nlp.dt - dq * nlp.dt

                external_forces = DynamicsFunctions.get_external_forces_from_contacts(
                    nlp, q, qdot, nlp.model.contact_types, external_forces
                )
                # TODO: We do not use DynamicsFunctions.inverse_dynamics here since tau is not in the variables (this should be refactored)
                tau_id = nlp.model.inverse_dynamics(with_contact=False)(
                    q, qdot, slope_qdot, external_forces, nlp.parameters.cx
                )
                tau_defects = tau - tau_id
                defects[nlp.states["qdot"].index, 0] = tau_defects

                if has_excitation:
                    mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
                    dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations, mus_activations)
                    slope_mus = nlp.states_dot["muscles"].cx
                    defects[nlp.states["muscles"].index, 0] = slope_mus * nlp.dt - dmus * nlp.dt

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


class JointAccelerationDynamics(AbstractModel):
    """
    This class is used to create a model actuated through joint acceleration.

    x = [q, qdot]
    u = [qddot_joints]
    """

    def __init__(self):
        super().__init__()
        self.state_configuration = [States.Q, States.QDOT]
        self.control_configuration = [Controls.QDDOT_JOINTS]

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

        # Get variables from the right place
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        qddot_joints = DynamicsFunctions.get(nlp.controls["qddot_joints"], controls)

        qddot_root = nlp.model.forward_dynamics_free_floating_base()(q, qdot, qddot_joints, nlp.parameters.cx)
        qddot_reordered = nlp.model.reorder_qddot_root_joints(qddot_root, qddot_joints)

        qdot_mapped = nlp.variable_mappings["qdot"].to_first.map(qdot)
        qddot_mapped = nlp.variable_mappings["qdot"].to_first.map(qddot_reordered)

        dxdt = nlp.cx(nlp.states.shape, 1)
        dxdt[q_indices, 0] = qdot_mapped
        dxdt[qdot_indices, 0] = qddot_mapped

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):

            DynamicsFunctions.no_states_mapping(nlp)
            slope_q, slope_qdot = self.get_basic_slopes(nlp)

            # Initialize defects
            defects = nlp.cx(nlp.states.shape, 1)

            # qdot = polynomial slope
            defects[q_indices, 0] = slope_q * nlp.dt - qdot_mapped * nlp.dt

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                defects[qdot_indices, 0] = slope_qdot * nlp.dt - qddot_mapped * nlp.dt

            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for joints acceleration driven dynamics."
                )

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    def get_rigid_contact_forces(self, time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp):
        raise RuntimeError("Joints acceleration driven dynamics cannot be used with contacts by definition.")


DynamicalModel: TypeAlias = (
    TorqueDynamics
    | StochasticTorqueDynamics
    | TorqueFreeFloatingBaseDynamics
    | StochasticTorqueFreeFloatingBaseDynamics
    | MusclesDynamics
    | TorqueActivationDynamics
    | TorqueDerivativeDynamics
    | JointAccelerationDynamics
    | HolonomicTorqueDynamics
)
