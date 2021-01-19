from typing import Union

import numpy as np
from casadi import MX
import biorbd


class BiorbdInterface:
    """
    Type conversion allowing to use biorbd with numpy arrays

    Methods
    -------
    convert_array_to_external_forces(all_f_ext: Union[list, tuple]) -> list[list[biorbd.VecBiorbdSpatialVector]]
        Convert external forces np.ndarray lists of external forces to values understood by biorbd
    """

    @staticmethod
    def convert_array_to_external_forces(all_f_ext: Union[list, tuple]) -> list[list[biorbd.VecBiorbdSpatialVector]]:
        """
        Convert external forces np.ndarray lists of external forces to values understood by biorbd

        Parameters
        ----------
        all_f_ext: Union[list, tuple]
            The external forces that acts on the model (the size of the matrix should be
            6 x number of external forces x number of shooting nodes OR 6 x number of shooting nodes)

        Returns
        -------
        The same forces in a biorbd-friendly format
        """

        if not isinstance(all_f_ext, (list, tuple)):
            raise RuntimeError(
                "f_ext should be a list of (6 x nb_external_forces x nb_shooting) or (6 x nb_shooting) matrix"
            )

        sv_over_all_phases = []
        for f_ext in all_f_ext:
            f_ext = np.array(f_ext)
            if len(f_ext.shape) < 2 or len(f_ext.shape) > 3:
                raise RuntimeError(
                    "f_ext should be a list of (6 x nb_external_forces x nb_shooting) or (6 x nb_shooting) matrix"
                )
            if len(f_ext.shape) == 2:
                f_ext = f_ext[:, :, np.newaxis]

            if f_ext.shape[0] != 6:
                raise RuntimeError(
                    "f_ext should be a list of (6 x nb_external_forces x nb_shooting) or (6 x nb_shooting) matrix"
                )

            sv_over_phase = []
            for node in range(f_ext.shape[2]):
                sv = biorbd.VecBiorbdSpatialVector()
                for idx in range(f_ext.shape[1]):
                    sv.append(biorbd.SpatialVector(MX(f_ext[:, idx, node])))
                sv_over_phase.append(sv)
            sv_over_all_phases.append(sv_over_phase)

        return sv_over_all_phases
