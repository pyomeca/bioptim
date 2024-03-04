from .objective_functions import ObjectiveFunction
from .phase_transition import PhaseTransition, PhaseTransitionFcn, PhaseTransitionList
from ..misc.enums import ControlType


class PhaseTransitionBuilder:
    """
    A class to prepare the phase transitions for the ocp builder

    Methods
    -------
    create_default_transitions()
        Create the default phase transitions for states continuity between phases.
    extend_transitions_for_linear_continuous()
        Add phase transitions for linear continuous controls.
    update_existing_transitions()
        Update the existing phase transitions with Mayer functions and add cyclic transitions

    Attributes
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    full_phase_transitions: list[PhaseTransition]
        The list of all the transitions prepared
    """

    def __init__(self, ocp):
        self.ocp = ocp
        self.full_phase_transitions = self.create_default_transitions()

    def create_default_transitions(self) -> list[PhaseTransition]:
        """Create the default phase transitions for states continuity between phases."""
        return [
            PhaseTransition(
                phase_pre_idx=i,
                transition=PhaseTransitionFcn.CONTINUOUS,
                weight=self.ocp.nlp[i].dynamics_type.state_continuity_weight,
            )
            for i in range(self.ocp.n_phases - 1)
        ]

    def extend_transitions_for_linear_continuous_controls(self):
        """Add phase transitions for linear continuous controls.
        This is a special case where the controls are continuous"""
        for phase, nlp in enumerate(self.ocp.nlp):
            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                self.full_phase_transitions.extend(
                    [
                        PhaseTransition(
                            phase_pre_idx=i,
                            transition=PhaseTransitionFcn.CONTINUOUS_CONTROLS,
                            weight=None,
                        )
                        for i in range(self.ocp.n_phases - 1)
                    ]
                )

    def check_phase_index(self, idx_phase):
        """Check if the phase index is valid."""
        if idx_phase >= self.ocp.n_phases:
            raise RuntimeError("Phase index of the phase transition is higher than the number of phases")

    def update_transition_base(self, pt):
        """Update the transition base with Mayer functions
        if the user provided a weight like for an objective function."""
        if pt.weight:
            pt.base = ObjectiveFunction.MayerFunction

    def handle_cyclic_transition(self, idx_phase, pt):
        """The case of a cyclic transition, the terminal phase is linked (n) to the initial phase (0)."""
        if idx_phase % self.ocp.n_phases == self.ocp.n_phases - 1:
            self.full_phase_transitions.append(pt)
        else:
            self.full_phase_transitions[idx_phase] = pt

    def update_existing_transitions(self, phase_transition_list) -> list[PhaseTransition]:
        """Update the existing phase transitions with Mayer functions and add cyclic transitions."""
        existing_phases = []
        for pt in phase_transition_list:
            idx_phase = pt.nodes_phase[0]
            self.check_phase_index(idx_phase)
            existing_phases.append(idx_phase)
            self.update_transition_base(pt)
            self.handle_cyclic_transition(idx_phase, pt)
        return self.full_phase_transitions

    def prepare_phase_transitions(self, phase_transition_list: PhaseTransitionList) -> list[PhaseTransition]:
        """
        Configure all the phase transitions and put them in a list

        Parameters
        ----------
        phase_transition_list: PhaseTransitionList
            The phase transitions to prepare added by the user

        Returns
        -------
        list[PhaseTransition]
            The list of all the transitions prepared
        """
        self.extend_transitions_for_linear_continuous_controls()
        return self.update_existing_transitions(phase_transition_list)
