from typing import Any, Callable

from casadi import MX, SX, vertcat

from ..optimization.non_linear_program import NonLinearProgram
from ..optimization.optimization_variable import OptimizationVariableList
from ..misc.enums import ControlType, Node


class PenaltyController:
    """
    A placeholder for the required elements to compute a penalty (all time)
    For most of the part, PenaltyController will behave like NLP with the major difference that it will always
    return the states and controls form the current node_index (instead of all of them). If for some reason,
    one must access specific node that is not the current node_index, they can directly access the _nlp.
    Please note that this will likely result in free variables, which can be a pain to deal with...
    """

    def __init__(
        self,
        ocp,
        nlp: NonLinearProgram,
        t: list,
        x: list,
        u: list,
        x_scaled: list,
        u_scaled: list,
        p: MX | SX | list,
        s: list,
        node_index: int = None,
    ):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        t: list
            Time indices, maximum value being the number of shooting point + 1
        x: list
            References to the state variables
        u: list
            References to the control variables
        x_scaled: list
            References to the scaled state variables
        u_scaled: list
            References to the scaled control variables
        p: MX | SX | list
            References to the parameter variables
        node_index: int
            Current node index if ocp.assume_phase_dynamics is True, then node_index is expected to be set to 0
        """

        self._ocp: Any = ocp
        self._nlp: NonLinearProgram = nlp
        self.t = t
        self.x = x
        self.u = u
        self.x_scaled = x_scaled
        self.u_scaled = u_scaled
        self.p = vertcat(p) if p is not None else p
        self.node_index = node_index
        self.cx_index_to_get = 0

    def __len__(self):
        return len(self.t)

    @property
    def ocp(self):
        return self._ocp

    @property
    def get_nlp(self):
        """
        This method returns the underlying nlp. Please note that acting directly with the nlp is not want you should do.
        Unless you see no way to access what you need otherwise, we strongly suggest that you use the normal path
        """
        return self._nlp

    @property
    def cx(self) -> MX | SX | Callable:
        return self._nlp.cx

    @property
    def to_casadi_func(self) -> Callable:
        return self._nlp.to_casadi_func

    @property
    def control_type(self) -> ControlType:
        return self._nlp.control_type

    @property
    def ode_solver(self) -> ControlType:
        return self._nlp.ode_solver

    @property
    def phase_idx(self) -> int:
        return self._nlp.phase_idx

    @property
    def ns(self) -> int:
        return self._nlp.ns

    @property
    def tf(self) -> int:
        return self._nlp.tf

    @property
    def mx_to_cx(self):
        return self._nlp.mx_to_cx

    @property
    def model(self):
        return self._nlp.model

    @property
    def states(self) -> OptimizationVariableList:
        """
        Return the states associated with the current node index

        Returns
        -------
        The states at node node_index
        """
        self._nlp.states.node_index = self.node_index
        out = self._nlp.states.unscaled
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def controls(self) -> OptimizationVariableList:
        """
        Return the controls associated with the current node index

        Returns
        -------
        The controls at node node_index
        """
        self._nlp.controls.node_index = self.node_index
        out = self._nlp.controls.unscaled
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def states_dot(self) -> OptimizationVariableList:
        """
        Return the states_dot associated with the current node index

        Returns
        -------
        The states_dot at node node_index
        """
        self._nlp.states_dot.node_index = self.node_index
        out = self._nlp.states_dot.unscaled
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def stochastic_variables(self) -> OptimizationVariableList:
        """
        Return the stochastic_variables associated with the current node index
        Returns
        -------
        The stochastic_variables at node node_index
        """
        # TODO: This variables should be scaled and renamed to "algebraic"
        self._nlp.stochastic_variables.node_index = self.node_index
        out = self._nlp.stochastic_variables
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def integrated_values(self) -> OptimizationVariableList:
        """
        Return the values associated with the current node index
        Returns
        -------
        The integrated_values at node node_index
        """
        self._nlp.integrated_values.node_index = self.node_index
        out = self._nlp.integrated_values
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def motor_noise(self):
        return self._nlp.motor_noise

    @property
    def sensory_noise(self):
        return self._nlp.sensory_noise

    @property
    def integrate(self):
        return self._nlp.dynamics[self.node_index]

    @property
    def integrate_noised_dynamics(self):
        return self._nlp.noised_dynamics[self.node_index]

    @property
    def dynamics(self):
        return self._nlp.dynamics_func

    @property
    def states_scaled(self) -> OptimizationVariableList:
        """
        Return the scaled states associated with the current node index.

        Warning: Most of the time, the user does not want that states but the normal `states`, that said, it can
        sometime be useful for very limited number of use case.

        Returns
        -------
        The scaled states at node node_index
        """
        self._nlp.states.node_index = self.node_index
        out = self._nlp.states.scaled
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def controls_scaled(self) -> OptimizationVariableList:
        """
        Return the scaled controls associated with the current node index.

        Warning: Most of the time, the user does not want that controls but the normal `controls`, that said, it can
        sometime be useful for very limited number of use case.

        Returns
        -------
        The scaled controls at node node_index
        """
        self._nlp.controls.node_index = self.node_index
        out = self._nlp.controls.scaled
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def states_dot_scaled(self) -> OptimizationVariableList:
        """
        Return the scaled states_dot associated with the current node index.

        Warning: Most of the time, the user does not want that states but the normal `states_dot`, that said, it can
        sometime be useful for very limited number of use case.

        Returns
        -------
        The scaled states_dot at node node_index
        """
        self._nlp.states_dot.node_index = self.node_index

        out = self._nlp.states_dot.scaled
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def parameters(self) -> OptimizationVariableList:
        """
        Return the parameters

        Returns
        -------
        The parameters
        """
        return self._nlp.parameters

    @property
    def parameters_scaled(self) -> OptimizationVariableList:
        """
        Return the scaled parameters

        Warning: Most of the time, the user does not want that parameters but the normal `parameters`, that said, it can
        sometime be useful for very limited number of use case.

        Returns
        -------
        The scaled parameters
        """
        return self._nlp.parameters.scaled
