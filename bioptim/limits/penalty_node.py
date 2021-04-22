from typing import Union, Any

from casadi import MX, SX, vertcat

from ..optimization.non_linear_program import NonLinearProgram


class PenaltyNodes:
    """
    A placeholder for the required elements to compute a penalty (all time)
    """

    def __init__(self, ocp, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX, list]):
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
        p: Union[MX, SX]
            References to the parameter variables
        """

        self.ocp: Any = ocp
        self.nlp: NonLinearProgram = nlp
        self.t = t
        self.x = x
        self.u = u
        self.p = vertcat(p)
