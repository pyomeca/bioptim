from typing import Union, Callable, Any

import numpy as np
from casadi import MX, SX, Function
import biorbd_casadi as biorbd


class BiorbdInterface:
    """
    Type conversion allowing to use biorbd with numpy arrays

    Methods
    -------
    convert_array_to_external_forces(all_f_ext: Union[list, tuple]) -> list[list[biorbd.VecBiorbdSpatialVector]]
        Convert external forces np.ndarray lists of external forces to values understood by biorbd
    mx_to_cx(name: str, function: Union[Callable, SX, MX], *all_param: Any) -> Function
        Add to the pool of declared casadi function. If the function already exists, it is skipped
    """

    @staticmethod
    def convert_array_to_external_forces(all_f_ext: Union[list, tuple]) -> list:
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
                "f_ext should be a list of (6 x n_external_forces x n_shooting) or (6 x n_shooting) matrix"
            )

        sv_over_all_phases = []
        for f_ext in all_f_ext:
            f_ext = np.array(f_ext) if isinstance(f_ext, list) else f_ext
            if len(f_ext.shape) < 2 or len(f_ext.shape) > 3:
                raise RuntimeError(
                    "f_ext should be a list of (6 x n_external_forces x n_shooting) or (6 x n_shooting) matrix"
                )
            if len(f_ext.shape) == 2 and type(f_ext).__module__ == "numpy":
                f_ext = f_ext[:, :, np.newaxis]
                size_3rd_dimension = f_ext.shape[2]

            elif len(f_ext.shape) == 2 and type(f_ext).__module__ == "casadi.casadi":
                size_3rd_dimension = 1
            else:
                size_3rd_dimension = f_ext.shape[2]

            if f_ext.shape[0] != 6:
                raise RuntimeError(
                    "f_ext should be a list of (6 x n_external_forces x n_shooting) or (6 x n_shooting) matrix"
                )

            sv_over_phase = []
            for node in range(size_3rd_dimension):
                sv = biorbd.VecBiorbdSpatialVector()
                for idx in range(f_ext.shape[1]):
                    if type(f_ext).__module__ == "casadi.casadi":
                        sv.append(biorbd.SpatialVector(f_ext[:, idx]))
                    elif type(f_ext).__module__ == "numpy":
                        sv.append(biorbd.SpatialVector(MX(f_ext[:, idx, node])))
                    else:
                        raise NotImplementedError("This function is only implemented for list of numpy or casadi array")
                sv_over_phase.append(sv)
            sv_over_all_phases.append(sv_over_phase)

        return sv_over_all_phases

    @staticmethod
    def mx_to_cx(name: str, function: Union[Callable, SX, MX], *all_param: Any) -> Function:
        """
        Add to the pool of declared casadi function. If the function already exists, it is skipped

        Parameters
        ----------
        name: str
            The unique name of the function to add to the casadi functions pool
        function: Union[Callable, SX, MX]
            The biorbd function to add
        all_param: Any
            Any parameters to pass to the biorbd function
        """
        from ..optimization.optimization_variable import OptimizationVariable, OptimizationVariableList
        from ..optimization.parameters import Parameter, ParameterList

        cx_types = OptimizationVariable, OptimizationVariableList, Parameter, ParameterList
        mx = [var.mx if isinstance(var, cx_types) else var for var in all_param]
        cx = [var.mapping.to_second.map(var.cx) for var in all_param if isinstance(var, cx_types)]
        return biorbd.to_casadi_func(name, function, *mx)(*cx)
