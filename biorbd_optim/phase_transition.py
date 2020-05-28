class PhaseTransition:
    @staticmethod
    def continuous(ocp, phase_before_idx):
        """
        TODO
        """
        return ocp.nlp[phase_before_idx]["X"][-1] - ocp.nlp[phase_before_idx + 1]["X"][0]

    @staticmethod
    def impact(ocp, phase_before_idx):
        """
        TODO
        """
        pass
