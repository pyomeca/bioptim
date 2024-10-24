from typing import Any

import numpy as np

from ..misc.options import OptionGeneric, OptionDict
from ..misc.enums import ExternalForcesType, ReferenceFrame

class ExternalForces(OptionGeneric):

    def __init__(
        self,
        key: str,
        linear_force_data: np.ndarray,
        torque_data: np.ndarray,
        force_reference_frame: ReferenceFrame,
        point_of_application_reference_frame: ReferenceFrame,
        point_of_application: np.ndarray = None,
        **extra_parameters: Any,
    ):
        super(ExternalForces, self).__init__(**extra_parameters)
        if self.list_index != -1 and self.list_index is not None:
            raise NotImplementedError("All external forces must be declared, list_index cannot be used for now.")

        if linear_force_data is not None and linear_force_data.shape[0] != 3:
            raise ValueError(f"External forces must have 3 rows, got {linear_force_data.shape[0]}")

        if torque_data is not None and torque_data.shape[0] != 3:
            raise ValueError(f"External torques must have 3 rows, got {torque_data.shape[0]}")

        self.key = key
        self.linear_force_data = linear_force_data
        self.torque_data = torque_data
        self.force_reference_frame = force_reference_frame
        self.point_of_application = point_of_application
        self.point_of_application_reference_frame = point_of_application_reference_frame

    @property
    def len(self):
        """
        Returns the number of nodes in the external forces
        """
        if self.linear_force_data is not None:
            return self.linear_force_data.shape[1]
        elif self.torque_data is not None:
            return self.torque_data.shape[1]
        else:
            raise ValueError("External forces must have either linear_force_data or torque_data defined")

class ExternalForcesList(OptionDict):

    def __init__(self, *args, **kwargs):
        super(ExternalForcesList, self).__init__(sub_type=ExternalForces)

    def add(
        self,
        key: str,
        phase: int,
        data: np.ndarray,
        force_type: ExternalForcesType,
        force_reference_frame: ReferenceFrame,
        point_of_application: np.ndarray = None,
        point_of_application_reference_frame: ReferenceFrame = None,
        **extra_arguments: Any,
    ):

        if key in self and self[key].phase == phase and self[key].force_type == force_type:
            raise ValueError(
                f"There should be only one external force with the same key, force_type and phase (key:{key}, force_type: {force_type}, phase:{phase} already exists)")

        if force_reference_frame not in [ReferenceFrame.GLOBAL, ReferenceFrame.LOCAL]:
            raise ValueError(
                f"External force reference frame must be of type ReferenceFrame.GLOBAL or ReferenceFrame.LOCAL, got{force_reference_frame}")

        if point_of_application_reference_frame not in [ReferenceFrame.GLOBAL, ReferenceFrame.LOCAL, None]:
            raise ValueError(
                f"Point of application reference frame must be of type ReferenceFrame.GLOBAL or ReferenceFrame.LOCAL, got{point_of_application_reference_frame}")

        if point_of_application is not None:
            if point_of_application.shape[0] != 3:
                raise ValueError(f"Point of application must have 3 rows, got {point_of_application.shape[0]}")
            if force_type == ExternalForcesType.TORQUE:
                raise ValueError("Point of application cannot be used with ExternalForcesType.TORQUE")

        if force_type == ExternalForcesType.TORQUE and force_reference_frame != ReferenceFrame.GLOBAL:
            raise ValueError("External torques are defined in global reference frame")

        if force_type == ExternalForcesType.TORQUE and key in self.keys():
            # Do not change the reference frame of the linear force
            force_reference_frame = self[key].force_reference_frame

        linear_force_data = None
        if key in self.keys() and self[key].linear_force_data is not None:
            if force_type == ExternalForcesType.LINEAR_FORCE:
                raise ValueError(f"The linear force is already defined for {key}")
            else:
                linear_force_data = self[key].linear_force_data
        elif force_type == ExternalForcesType.LINEAR_FORCE:
            linear_force_data = data

        torque_data = None
        if key in self.keys() and self[key].torque_data is not None:
            if force_type == ExternalForcesType.TORQUE:
                raise ValueError(f"The torque is already defined for {key}")
            else:
                torque_data = self[key].torque_data
        elif force_type == ExternalForcesType.TORQUE:
            torque_data = data

        super(ExternalForcesList, self)._add(
            key=key,
            phase=phase,
            linear_force_data=linear_force_data,
            torque_data=torque_data,
            force_reference_frame=force_reference_frame,
            point_of_application=point_of_application,
            point_of_application_reference_frame=point_of_application_reference_frame,
            **extra_arguments,
        )

    def print(self):
        raise NotImplementedError("Printing of ExternalForcesList is not ready yet")

def get_external_forces_segments(external_forces: ExternalForcesList):

    segments_to_apply_forces_in_global = []
    segments_to_apply_forces_in_local = []
    segments_to_apply_translational_forces = []
    if external_forces is not None:
        for key in external_forces.keys():
            force = external_forces[key]
            # Check sanity first
            if force.force_reference_frame == ReferenceFrame.GLOBAL and force.point_of_application_reference_frame == ReferenceFrame.LOCAL and force.torque_data is not None:
                raise NotImplementedError("External forces in global reference frame cannot have a point of application in the local reference frame and torques defined at the same time yet")
            elif force.force_reference_frame == ReferenceFrame.LOCAL and force.point_of_application_reference_frame == ReferenceFrame.GLOBAL:
                raise NotImplementedError("External forces in local reference frame cannot have a point of application in the global reference frame yet")

            if force.force_reference_frame == ReferenceFrame.GLOBAL and (force.point_of_application_reference_frame == ReferenceFrame.GLOBAL or force.point_of_application_reference_frame is None):
                segments_to_apply_forces_in_global.append(force.key)
            elif force.force_reference_frame == ReferenceFrame.LOCAL and (force.point_of_application_reference_frame == ReferenceFrame.LOCAL or force.point_of_application_reference_frame is None):
                segments_to_apply_forces_in_local.append(force.key)
            elif force.force_reference_frame == ReferenceFrame.GLOBAL and force.point_of_application_reference_frame == ReferenceFrame.LOCAL:
                segments_to_apply_translational_forces.append(force.key)
            else:
                raise ValueError("This should not happen")

    return segments_to_apply_forces_in_global, segments_to_apply_forces_in_local, segments_to_apply_translational_forces