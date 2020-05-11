import numpy as np
from casadi import MX
import biorbd


class BiorbdInterface:
    @staticmethod
    def convert_array_to_external_forces(all_f_ext):
        # TODO: Look if f_ext is already what is needed

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
