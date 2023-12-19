from enum import Enum, auto
import numpy as np


class SolutionMerge(Enum):
    """
    The level of merging that can be done on the solution
    """

    KEYS = auto()
    NODES = auto()
    PHASES = auto()
    ALL = auto()


class SolutionData:
    def __init__(self, unscaled, scaled, n_nodes: list[int, ...]):
        """
        Parameters
        ----------
        ... # TODO
        n_nodes: list
            The number of node at each phase
        """
        self.unscaled = unscaled
        self.scaled = scaled
        self.n_phases = len(self.unscaled)
        self.n_nodes = n_nodes  # This is painfully necessary to get from outside to merge key if no keys are available

    @staticmethod
    def from_unscaled(ocp, unscaled: list, variable_type: str):
        """
        Create a SolutionData from unscaled values

        Parameters
        ----------
        ocp
            A reference to the ocp
        unscaled: list
            The unscaled values
        variable_type: str
            The type of variable to convert (x for states, u for controls, p for parameters, s for stochastic variables)
        """
        n_nodes = [nlp.n_states_nodes for nlp in ocp.nlp]
        return SolutionData(unscaled, _to_scaled_values(unscaled, ocp, variable_type), n_nodes)

    @staticmethod
    def from_scaled(ocp, scaled: list, variable_type: str):
        """
        Create a SolutionData from scaled values

        Parameters
        ----------
        ocp
            A reference to the ocp
        scaled: list
            The scaled values
        variable_type: str
            The type of variable to convert (x for states, u for controls, p for parameters, s for stochastic variables)
        """
        n_nodes = [nlp.n_states_nodes for nlp in ocp.nlp]
        return SolutionData(_to_unscaled_values(scaled, ocp, variable_type), scaled, n_nodes)

    def __getitem__(self, **keys):
        phase = 0
        if len(self.unscaled) > 1:
            if "phase" not in keys:
                raise RuntimeError("You must specify the phase when more than one phase is present in the solution")
            phase = keys["phase"]
        key = keys["key"]
        return self.unscaled[phase][key]

    def keys(self, phase: int = 0):
        return self.unscaled[phase].keys()

    def to_dict(self, to_merge: SolutionMerge | list[SolutionMerge, ...] = None, scaled: bool = False):
        data = self.scaled if scaled else self.unscaled
        
        if to_merge is None:
            to_merge = []

        if isinstance(to_merge, SolutionMerge):
            to_merge = [to_merge]

        if SolutionMerge.ALL in to_merge:
            to_merge = [SolutionMerge.KEYS, SolutionMerge.NODES, SolutionMerge.PHASES]

        if not to_merge:
            return data

        # Before merging phases, we must go inside the phases
        out = []
        for phase_idx in range(len(data)):
            if SolutionMerge.KEYS in to_merge and SolutionMerge.NODES in to_merge:
                phase_data = self._merge_keys_nodes(data, phase=phase_idx)
            elif SolutionMerge.KEYS in to_merge:
                phase_data = self._merge_keys(data, phase=phase_idx)
            elif SolutionMerge.NODES in to_merge:
                phase_data = self._merge_nodes(data, phase=phase_idx)
            else:
                raise ValueError("Merging must at least contain SolutionMerge.KEYS or SolutionMerge.NODES")

            out.append(phase_data)

        if SolutionMerge.PHASES in to_merge:
            out = self._merge_phases(out, to_merge=to_merge)
            
        return out

    @staticmethod
    def _merge_phases(data: list, to_merge: list[SolutionMerge, ...]):
        """
        Merge the phases by merging keys and nodes before. 
        This method does not remove the redundent nodes when merging the phase nor the nodes
        """
        
        if SolutionMerge.NODES not in to_merge:
            raise ValueError("to_merge must contain SolutionMerge.NODES when merging phases")
        if SolutionMerge.KEYS not in to_merge:
            return {
                key: np.concatenate([data[phase][key] for phase in range(len(data))], axis=1) for key in data[0].keys()
            }

        return np.concatenate(data, axis=1)

    def _merge_keys_nodes(self, data: dict, phase: int):
        """
        Merge the steps by merging keys before.
        """
        return np.concatenate(self._merge_keys(data, phase=phase), axis=1)
    
    @staticmethod
    def _merge_nodes(data: dict, phase: int):
        """
        Merge the nodes of a SolutionData.unscaled or SolutionData.scaled, without merging the keys

        Returns
        -------
        The merged data
        """

        return {key: np.concatenate(data[phase][key], axis=1) for key in data[phase].keys()}

    def _merge_keys(self, data: dict, phase: int):
        """
        Merge the keys without merging anything else
        """

        if not data[phase].keys():
            return [np.ndarray((0, 1))] * self.n_nodes[phase]
        
        n_nodes = len(data[phase][list(data[phase].keys())[0]])
        out = []
        for node_idx in range(n_nodes):
            out.append(np.concatenate([data[phase][key][node_idx] for key in data[phase].keys()], axis=0))
        return out


def _to_unscaled_values(scaled: list, ocp, variable_type: str) -> list:
    """
    Convert values of scaled solution to unscaled values

    Parameters
    ----------
    scaled: list
        The scaled values
    variable_type: str
        The type of variable to convert (x for states, u for controls, p for parameters, s for stochastic variables)
    """

    unscaled: list = [None for _ in range(len(scaled))]
    for phase in range(len(scaled)):
        unscaled[phase] = {}
        for key in scaled[phase].keys():
            if variable_type == "p":
                scale_factor = ocp.parameters[key].scaling
            else:
                scale_factor = getattr(ocp.nlp[phase], f"{variable_type}_scaling")[key]

            if isinstance(scaled[phase][key], list):  # Nodes are not merged
                unscaled[phase][key] = []
                for node in range(len(scaled[phase][key])):
                    value = scaled[phase][key][node]
                    unscaled[phase][key].append(value * scale_factor.to_array(value.shape[1]))
            elif isinstance(scaled[phase][key], np.ndarray):  # Nodes are merged
                value = scaled[phase][key]
                unscaled[phase][key] = value * scale_factor.to_array(value.shape[1])
            else:
                raise ValueError(f"Unrecognized type {type(scaled[phase][key])} for {key}")

    return unscaled


def _to_scaled_values(unscaled: list, ocp, variable_type: str) -> list:
    """
    Convert values of unscaled solution to scaled values

    Parameters
    ----------
    unscaled: list
        The unscaled values
    variable_type: str
        The type of variable to convert (x for states, u for controls, p for parameters, s for stochastic variables)
    """

    if not unscaled:
        return []

    scaled: list = [None for _ in range(len(unscaled))]
    for phase in range(len(unscaled)):
        scaled[phase] = {}
        for key in unscaled[phase].keys():
            if variable_type == "p":
                scale_factor = ocp.parameters[key].scaling
            else:
                scale_factor = getattr(ocp.nlp[phase], f"{variable_type}_scaling")[key]

            if isinstance(unscaled[phase][key], list):  # Nodes are not merged
                scaled[phase][key] = []
                for node in range(len(unscaled[phase][key])):
                    value = unscaled[phase][key][node]
                    scaled[phase][key].append(value / scale_factor.to_array(value.shape[1]))
            elif isinstance(unscaled[phase][key], np.ndarray):  # Nodes are merged
                value = unscaled[phase][key]
                scaled[phase][key] = value / scale_factor.to_array(value.shape[1])
            else:
                raise ValueError(f"Unrecognized type {type(unscaled[phase][key])} for {key}")

    return scaled


