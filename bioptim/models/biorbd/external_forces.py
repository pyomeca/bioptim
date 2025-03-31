import numpy as np
from casadi import MX, vertcat


class ExternalForceSetTimeSeries:
    """
    A class to manage external forces applied to a set of segments over a series of frames.

    Attributes
    ----------
    _nb_frames : int
        The number of frames in the time series.
    in_global : dict[str, {}]
        Dictionary to store global external forces for each segment.
    torque_in_global : dict[str, {}]
        Dictionary to store global torques for each segment.
    translational_in_global : dict[str, {}]
        Dictionary to store global translational forces for each segment.
    in_local : dict[str, {}]
        Dictionary to store local external forces for each segment.
    torque_in_local : dict[str, {}]
        Dictionary to store local torques for each segment.
    _bind : bool
        Flag to indicate if the external forces are binded and cannot be modified.
    """

    def __init__(self, nb_frames: int):
        """
        Initialize the ExternalForceSetTimeSeries with the number of frames.

        Parameters
        ----------
        nb_frames : int
            The number of frames in the time series.
        """
        self._nb_frames = nb_frames

        self.in_global: dict[str, {}] = {}
        self.torque_in_global: dict[str, {}] = {}
        self.translational_in_global: dict[str, {}] = {}
        self.in_local: dict[str, {}] = {}
        self.torque_in_local: dict[str, {}] = {}

        self._bind_flag = False

    @property
    def _can_be_modified(self) -> bool:
        return not self._bind_flag

    def _check_if_can_be_modified(self) -> None:
        if not self._can_be_modified:
            raise RuntimeError("External forces have been binded and cannot be modified anymore.")

    @property
    def nb_frames(self) -> int:
        return self._nb_frames

    def _check_values_frame_shape(self, values: np.ndarray) -> None:
        if values.shape[1] != self._nb_frames:
            raise ValueError(
                f"External forces must have the same number of columns as the number of shooting points, "
                f"got {values.shape[1]} instead of {self._nb_frames}"
            )

    def add(self, segment: str, values: np.ndarray, point_of_application: np.ndarray | str = None):
        self._check_if_can_be_modified()
        if values.shape[0] != 6:
            raise ValueError(f"External forces must have 6 rows, got {values.shape[0]}")
        self._check_values_frame_shape(values)

        point_of_application = np.zeros((3, self._nb_frames)) if point_of_application is None else point_of_application
        self._check_point_of_application(point_of_application)

        self.in_global = ensure_list(self.in_global, segment)
        self.in_global[segment].append({"values": values, "point_of_application": point_of_application})

    def add_torque(self, segment: str, values: np.ndarray):
        self._check_if_can_be_modified()
        if values.shape[0] != 3:
            raise ValueError(f"External torques must have 3 rows, got {values.shape[0]}")
        self._check_values_frame_shape(values)

        self.torque_in_global = ensure_list(self.torque_in_global, segment)
        self.torque_in_global[segment].append({"values": values, "point_of_application": None})

    def add_translational_force(
        self, segment: str, values: np.ndarray, point_of_application_in_local: np.ndarray | str = None
    ):
        self._check_if_can_be_modified()
        if values.shape[0] != 3:
            raise ValueError(f"External forces must have 3 rows, got {values.shape[0]}")
        self._check_values_frame_shape(values)

        point_of_application_in_local = (
            np.zeros((3, self._nb_frames)) if point_of_application_in_local is None else point_of_application_in_local
        )
        self._check_point_of_application(point_of_application_in_local)
        self.translational_in_global = ensure_list(self.translational_in_global, segment)
        self.translational_in_global[segment].append(
            {"values": values, "point_of_application": point_of_application_in_local}
        )

    def add_in_segment_frame(
        self, segment: str, values: np.ndarray, point_of_application_in_local: np.ndarray | str = None
    ):
        """
        Add external forces in the segment frame.

        Parameters
        ----------
        segment: str
            The name of the segment.
        values: np.ndarray
            The external forces (torques, forces) in the segment frame.
        point_of_application_in_local
            The point of application of the external forces in the segment frame.
        """
        self._check_if_can_be_modified()
        if values.shape[0] != 6:
            raise ValueError(f"External forces must have 6 rows, got {values.shape[0]}")
        self._check_values_frame_shape(values)

        point_of_application_in_local = (
            np.zeros((3, self._nb_frames)) if point_of_application_in_local is None else point_of_application_in_local
        )
        self._check_point_of_application(point_of_application_in_local)
        self.in_local = ensure_list(self.in_local, segment)
        self.in_local[segment].append({"values": values, "point_of_application": point_of_application_in_local})

    def add_torque_in_segment_frame(self, segment: str, values: np.ndarray):
        self._check_if_can_be_modified()
        if values.shape[0] != 3:
            raise ValueError(f"External torques must have 3 rows, got {values.shape[0]}")
        self._check_values_frame_shape(values)

        self.torque_in_local = ensure_list(self.torque_in_local, segment)
        self.torque_in_local[segment].append({"values": values, "point_of_application": None})

    @property
    def nb_external_forces(self) -> int:
        attributes = ["in_global", "torque_in_global", "translational_in_global", "in_local", "torque_in_local"]
        return sum([len(values) for attr in attributes for values in getattr(self, attr).values()])

    @property
    def nb_external_forces_components(self) -> int:
        """Return the number of vertical components of the external forces if concatenated in a unique vector"""
        attributes_no_point_of_application = ["torque_in_global", "torque_in_local"]
        attributes_six_components = ["in_global", "in_local"]

        components = 0
        for attr in attributes_no_point_of_application:
            for values in getattr(self, attr).values():
                components += 3 * len(values)

        for values in self.translational_in_global.values():
            nb_point_of_application_as_str = sum([isinstance(force["point_of_application"], str) for force in values])
            components += 6 * len(values) - 3 * nb_point_of_application_as_str

        for attr in attributes_six_components:
            for values in getattr(self, attr).values():
                nb_point_of_application_as_str = sum(
                    [isinstance(force["point_of_application"], str) for force in values]
                )
                components += 9 * len(values) - 3 * nb_point_of_application_as_str

        return components

    def _check_point_of_application(self, point_of_application: np.ndarray | str) -> None:
        if isinstance(point_of_application, str):
            # The point of application is a string, nothing to check yet
            return

        point_of_application = np.array(point_of_application)
        if point_of_application.shape[0] != 3 and point_of_application.shape[1] != 3:
            raise ValueError(
                f"Point of application must have"
                f" 3 rows and {self._nb_frames} columns, got {point_of_application.shape}"
            )

        return

    def _check_segment_names(self, segment_names: tuple[str, ...]) -> None:
        attributes = ["in_global", "torque_in_global", "translational_in_global", "in_local", "torque_in_local"]
        wrong_segments = []
        for attr in attributes:
            for segment, _ in getattr(self, attr).items():
                if segment not in segment_names:
                    wrong_segments.append(segment)

        if wrong_segments:
            raise ValueError(
                f"Segments {wrong_segments} specified in the external forces are not in the model."
                f" Available segments are {segment_names}."
            )

    def _check_all_string_points_of_application(self, model_points_of_application) -> None:
        attributes = ["in_global", "translational_in_global", "in_local"]
        wrong_points_of_application = []
        for attr in attributes:
            for segment, forces in getattr(self, attr).items():
                for force in forces:
                    if (
                        isinstance(force["point_of_application"], str)
                        and force["point_of_application"] not in model_points_of_application
                    ):
                        wrong_points_of_application.append(force["point_of_application"])

        if wrong_points_of_application:
            raise ValueError(
                f"Points of application {wrong_points_of_application} specified in the external forces are not in the model."
                f" Available points of application are {model_points_of_application}."
            )

    def _bind(self):
        """prevent further modification of the external forces"""
        self._bind_flag = True

    def to_numerical_time_series(self):
        """Convert the external forces to a numerical time series"""
        fext_numerical_time_series = np.zeros((self.nb_external_forces_components, 1, self.nb_frames + 1))

        # "type of external force": (function to call, number of force components, number of point of application components)
        bioptim_to_vector_map = {
            "in_global": 6,
            "torque_in_global": 3,
            "translational_in_global": 3,
            "in_local": 6,
            "torque_in_local": 3,
        }

        symbolic_counter = 0
        for attr in bioptim_to_vector_map.keys():
            for segment, forces in getattr(self, attr).items():
                for force in forces:
                    array_point_of_application = isinstance(force["point_of_application"], np.ndarray)

                    start = symbolic_counter
                    stop = symbolic_counter + bioptim_to_vector_map[attr]
                    force_slicer = slice(start, stop)
                    fext_numerical_time_series[force_slicer, 0, :-1] = force["values"]

                    if array_point_of_application:
                        poa_slicer = slice(stop, stop + 3)
                        fext_numerical_time_series[poa_slicer, 0, :-1] = force["point_of_application"]

                    symbolic_counter = stop + 3 if array_point_of_application else stop

        return fext_numerical_time_series


def ensure_list(data, key) -> dict[str, list]:
    """Ensure that the key exists in the data and the value is a list"""
    if data.get(key) is None:
        data[key] = []
    return data


# Specific functions for adding each force type to improve readability
def _add_global_force(biorbd_external_forces, segment, force, point_of_application):
    biorbd_external_forces.add(segment, force, point_of_application)


def _add_torque_global(biorbd_external_forces, segment, torque, _):
    biorbd_external_forces.add(segment, vertcat(torque, MX([0, 0, 0])), MX([0, 0, 0]))


def _add_translational_global(biorbd_external_forces, segment, force, point_of_application):
    biorbd_external_forces.addTranslationalForce(force, segment, point_of_application)


def _add_local_force(biorbd_external_forces, segment, force, point_of_application):
    biorbd_external_forces.addInSegmentReferenceFrame(segment, force, point_of_application)


def _add_torque_local(biorbd_external_forces, segment, torque, _):
    biorbd_external_forces.addInSegmentReferenceFrame(segment, vertcat(torque, MX([0, 0, 0])), MX([0, 0, 0]))
