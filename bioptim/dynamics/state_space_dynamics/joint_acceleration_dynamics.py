from ..configure_variables import States, Controls
from ..dynamics_functions import DynamicsFunctions
from ..dynamics_evaluation import DynamicsEvaluation
from ..ode_solvers import OdeSolver
from ...misc.enums import DefectType
from .abstract_dynamics import StateDynamics


class JointAccelerationDynamics(StateDynamics):
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
            defects[q_indices, 0] = slope_q - qdot_mapped

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                defects[qdot_indices, 0] = slope_qdot - qddot_mapped

            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for joints acceleration driven dynamics."
                )

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)
