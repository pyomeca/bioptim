from typing import Any

from casadi import MX, SX, vertcat

from ..optimization.non_linear_program import NonLinearProgram
from ..optimization.optimization_variable import OptimizationVariableContainer


class PenaltyController:
    """
    A placeholder for the required elements to compute a penalty (all time)
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
        """

        self._ocp: Any = ocp
        self._nlp: NonLinearProgram = nlp
        self.t = t
        self.x = x
        self.u = u
        self.x_scaled = x_scaled
        self.u_scaled = u_scaled
        self.p = vertcat(p) if p is not None else p

    @property
    def ocp(self):
        return self._ocp

    @property
    def nlp(self):
        return self._nlp

    def states(self, node_index: int = 0) -> OptimizationVariableContainer:
        """
        Return the states associated with a specific node

        Parameters
        ----------
        node_index
            The index of the node to request the states from

        Returns
        -------
        The states at node node_index
        """
        return self._nlp.states[node_index]

    def controls(self, node_index: int = 0) -> OptimizationVariableContainer:
        """
        Return the controls associated with a specific node

        Parameters
        ----------
        node_index
            The index of the node to request the controls from

        Returns
        -------
        The controls at node node_index
        """
        return self._nlp.controls[node_index]

    def __len__(self):
        return len(self.t)
