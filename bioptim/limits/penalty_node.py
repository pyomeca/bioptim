from typing import Any

from casadi import MX, SX, vertcat

from ..optimization.non_linear_program import NonLinearProgram


class PenaltyNodeList:
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

        self.ocp: Any = ocp
        self.nlp: NonLinearProgram = nlp
        self.t = t
        self.x = x
        self.u = u
        self.x_scaled = x_scaled
        self.u_scaled = u_scaled
        self.p = vertcat(p) if p is not None else p

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

    def __init__(self, nodes: PenaltyNodeList, shooting_index: int):
        """
        Parameters
        ----------
        nodes: PenaltyNodeList
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

    def __getitem__(self, item):
        variable_type = "any"
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str):
            variable_type = item[1]
            item = item[0]

        if isinstance(item, str):
            if item == "states":
                return self.x
            if item == "controls":
                return self.u

            if variable_type == "any":
                if item in self.nlp.states[0] and item in self.nlp.controls[0]: # TODO: [0] to [node_index]
                    raise RuntimeError(f"Sliced item must specify the type if they appear in both states and controls")

                if item in self.nlp.states[0]:  # TODO: [0] to [node_index]
                    return self.x[self.nlp.states[0][item].index, :]    # TODO: [0] to [node_index]
                elif item in self.nlp.controls[0]:  # TODO: [0] to [node_index]
                    return self.u[self.nlp.controls[0][item].index, :]  # TODO: [0] to [node_index]
                else:
                    raise RuntimeError(
                        f"{item} is not present in controls nor states. Or was not 'states' or 'controls', for all"
                    )

            elif variable_type == "states":
                return self.x[self.nlp.states[0][item].index, :]    # TODO: [0] to [node_index]

            elif variable_type == "controls":
                return self.u[self.nlp.controls[0][item].index, :]  # TODO: [0] to [node_index]

            else:
                raise ValueError("The variable_type must be 'any', 'states', or 'controls'")

        else:
            raise NotImplementedError("Slicing for penalty node is implemented only for str")
