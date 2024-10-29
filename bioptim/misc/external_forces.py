from typing import Any

import numpy as np

from ..misc.options import OptionGeneric, OptionDict
from ..misc.enums import ExternalForceType, ReferenceFrame

class ExternalForce(OptionGeneric):

    def __init__(
        self,
        key: str,
        force_data: np.ndarray,
        torque_data: np.ndarray,
        force_reference_frame: ReferenceFrame,
        point_of_application_reference_frame: ReferenceFrame,
        point_of_application: np.ndarray = None,
        **extra_parameters: Any,
    ):
        super(ExternalForce, self).__init__(**extra_parameters)
        if self.list_index != -1 and self.list_index is not None:
            raise NotImplementedError("All external forces must be declared, list_index cannot be used for now.")

        if force_data is not None and force_data.shape[0] != 3:
            raise ValueError(f"External forces must have 3 rows, got {force_data.shape[0]}")

        if torque_data is not None and torque_data.shape[0] != 3:
            raise ValueError(f"External torques must have 3 rows, got {torque_data.shape[0]}")

        self.key = key
        self.force_data = force_data
        self.torque_data = torque_data
        self.force_reference_frame = force_reference_frame
        self.point_of_application = point_of_application
        self.point_of_application_reference_frame = point_of_application_reference_frame

    @property
    def len(self):
        """
        Returns the number of nodes in the external forces
        """
        if self.force_data is not None:
            return self.force_data.shape[1]
        elif self.torque_data is not None:
            return self.torque_data.shape[1]
        else:
            raise ValueError("External forces must have either force_data or torque_data defined")

class ExternalForces(OptionDict):

    def __init__(self, *args, **kwargs):
        super(ExternalForces, self).__init__(sub_type=ExternalForce)

    def add(
        self,
        key: str,
        data: np.ndarray,
        force_type: ExternalForceType,
        force_reference_frame: ReferenceFrame,
        point_of_application: np.ndarray = None,
        point_of_application_reference_frame: ReferenceFrame = None,
        **extra_arguments: Any,
    ):

        if key in self and self[key].force_type == force_type:
            raise ValueError(
                f"There should be only one external force with the same key, force_type (key:{key} and force_type: {force_type} already exists)")

        if force_reference_frame not in [ReferenceFrame.GLOBAL, ReferenceFrame.LOCAL]:
            raise ValueError(
                f"External force reference frame must be of type ReferenceFrame.GLOBAL or ReferenceFrame.LOCAL, got{force_reference_frame}")

        if point_of_application_reference_frame not in [ReferenceFrame.GLOBAL, ReferenceFrame.LOCAL, None]:
            raise ValueError(
                f"Point of application reference frame must be of type ReferenceFrame.GLOBAL or ReferenceFrame.LOCAL, got{point_of_application_reference_frame}")

        if point_of_application is not None:
            if point_of_application.shape[0] != 3:
                raise ValueError(f"Point of application must have 3 rows, got {point_of_application.shape[0]}")
            if force_type == ExternalForceType.TORQUE:
                raise ValueError("Point of application cannot be used with ExternalForceType.TORQUE")

        if key in self.keys():
            # They must both be in the same reference frame
            if force_reference_frame != self[key].force_reference_frame:
                raise ValueError(f"External forces must be in the same reference frame, got {force_reference_frame} and {self[key].force_reference_frame}")

        force_data = None
        if key in self.keys() and self[key].force_data is not None:
            if force_type == ExternalForceType.FORCE:
                raise ValueError(f"The force is already defined for {key}")
            else:
                force_data = self[key].force_data
                point_of_application = self[key].point_of_application
                point_of_application_reference_frame = self[key].point_of_application_reference_frame
        elif force_type == ExternalForceType.FORCE:
            force_data = data

        torque_data = None
        if key in self.keys() and self[key].torque_data is not None:
            if force_type == ExternalForceType.TORQUE:
                raise ValueError(f"The torque is already defined for {key}")
            else:
                torque_data = self[key].torque_data
        elif force_type == ExternalForceType.TORQUE:
            torque_data = data

        if force_type == ExternalForceType.TORQUE_AND_FORCE:
            if force_data is not None:
                raise ValueError(f"The force is already defined for {key}")
            elif torque_data is not None:
                raise ValueError(f"The torque is already defined for {key}")
            else:
                torque_data = data[:3, :]
                force_data = data[3:, :]

        super(ExternalForces, self)._add(
            key=key,
            force_data=force_data,
            torque_data=torque_data,
            force_reference_frame=force_reference_frame,
            point_of_application=point_of_application,
            point_of_application_reference_frame=point_of_application_reference_frame,
            **extra_arguments,
        )

    def print(self):
        raise NotImplementedError("Printing of ExternalForces is not ready yet")

def get_external_forces_segments(external_forces: ExternalForces):

    segments_to_apply_forces_in_global = []
    segments_to_apply_forces_in_local = []
    segments_to_apply_translational_forces = []
    if external_forces is not None:
        for key in external_forces.keys():
            force_torsor = external_forces[key]
            # Check sanity first
            if force_torsor.force_reference_frame == ReferenceFrame.GLOBAL and force_torsor.point_of_application_reference_frame == ReferenceFrame.LOCAL and force.torque_data is not None:
                raise NotImplementedError("External forces in global reference frame cannot have a point of application in the local reference frame and torques defined at the same time yet")
            elif force_torsor.force_reference_frame == ReferenceFrame.LOCAL and force_torsor.point_of_application_reference_frame == ReferenceFrame.GLOBAL:
                raise NotImplementedError("External forces in local reference frame cannot have a point of application in the global reference frame yet")

            if force_torsor.force_reference_frame == ReferenceFrame.GLOBAL and (force_torsor.point_of_application_reference_frame == ReferenceFrame.GLOBAL or force_torsor.point_of_application_reference_frame is None):
                segments_to_apply_forces_in_global.append(force_torsor.key)
            elif force_torsor.force_reference_frame == ReferenceFrame.LOCAL and (force_torsor.point_of_application_reference_frame == ReferenceFrame.LOCAL or force_torsor.point_of_application_reference_frame is None):
                segments_to_apply_forces_in_local.append(force_torsor.key)
            elif force_torsor.force_reference_frame == ReferenceFrame.GLOBAL and force_torsor.point_of_application_reference_frame == ReferenceFrame.LOCAL:
                segments_to_apply_translational_forces.append(force_torsor.key)
            else:
                raise ValueError("This should not happen")

    return segments_to_apply_forces_in_global, segments_to_apply_forces_in_local, segments_to_apply_translational_forces