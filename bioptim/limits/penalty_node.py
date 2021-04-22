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

    def __len__(self):
        return len(self.t)

    def __iter__(self):
        """
        Allow for the list to be used in a for loop

        Returns
        -------
        A reference to self
        """

        self._iter_idx = 0
        return self

    def __next__(self):
        """
        Get the next phase of the option list

        Returns
        -------
        The next phase of the option list
        """

        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self[self._iter_idx - 1]

    def __getitem__(self, item):
        return PenaltyNode(self, item)


class PenaltyNode:
    """
    A placeholder for the required elements to compute a penalty (single time)
    """

    def __init__(self, nodes: PenaltyNodes, shooting_index: int):
        """
        Parameters
        ----------
        nodes: PenaltyNodes
            The penalty node for all the time
        shooting_index: int
            The index of the penalty node
        """

        self.ocp: Any = nodes.ocp
        self.nlp: NonLinearProgram = nodes.nlp
        self.t = nodes.t[shooting_index]
        self.x = nodes.x[shooting_index]
        self.u = nodes.u[shooting_index] if shooting_index < len(nodes.u) else None
        self.p = nodes.p
