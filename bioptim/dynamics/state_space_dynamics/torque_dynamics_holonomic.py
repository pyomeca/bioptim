from casadi import vertcat, DM

from ..configure_variables import States, Controls, AlgebraicStates
from ..dynamics_functions import DynamicsFunctions
from ..dynamics_evaluation import DynamicsEvaluation
from ..configure_variables import ConfigureVariables
from ..ode_solvers import OdeSolver
from ...misc.enums import DefectType
from .abstract_dynamics import StateDynamics


class HolonomicTorqueDynamics(StateDynamics):

    def __init__(self):
        super().__init__()
        self.state_configuration = [States.Q_U, States.QDOT_U]
        self.control_configuration = [Controls.TAU]
        self.algebraic_configuration = []
        self.functions = [
            ConfigureVariables.configure_qv,
            ConfigureVariables.configure_qdotv,
            ConfigureVariables.configure_lagrange_multipliers_function,
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
                defects = vertcat(slope_q, slope_qdot) - derivative
            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for holonomic torque driven dynamics."
                )

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    @property
    def extra_dynamics(self):
        return None
