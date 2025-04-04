import numpy as np
from casadi import MX, vertcat

from ...misc.parameters_types import Int, AnyDict, Bool, NpArray, Str, StrTuple, NpArrayOptional, CXOrDMOrNpArray


# "type of external force": (function to call, number of force components, number of point of application components)
BIOPTIM_TO_VECTOR_MAP = {
    "in_global": 6,
    "torque_in_global": 3,
    "translational_in_global": 3,
    "in_local": 6,
    "torque_in_local": 3,
}

class ExternalForceSetCommon:

    def __init__(self):
        self.in_global: dict[str, AnyDict] = {}
        self.torque_in_global: dict[str, AnyDict] = {}
        self.translational_in_global: dict[str, AnyDict] = {}
        self.in_local: dict[str, AnyDict] = {}
        self.torque_in_local: dict[str, AnyDict] = {}

        self._bind_flag = False

    @property
    def _can_be_modified(self) -> Bool:
        return not self._bind_flag

    def _check_if_can_be_modified(self) -> None:
        if not self._can_be_modified:
            raise RuntimeError("External forces have been binded and cannot be modified anymore.")

    def bind(self):
        """prevent further modification of the external forces"""
        self._bind_flag = True

    @property
    def nb_external_forces_components(self) -> int:
        """Return the number of vertical components of the external forces if concatenated in a unique vector"""
        attributes_no_point_of_application = ["torque_in_global", "torque_in_local"]
        attributes_three_components = ["translational_in_global"]
        attributes_six_components = ["in_global", "in_local"]

        components = 0
        for attr in attributes_no_point_of_application:
            for values in getattr(self, attr).values():
                components += 3

        for attr in attributes_three_components:
            for values in getattr(self, attr).values():
                is_point_of_application_str = isinstance(values["point_of_application"], str)
                components += 6 - 3 * is_point_of_application_str

        for attr in attributes_six_components:
            for values in getattr(self, attr).values():
                is_point_of_application_str = isinstance(values["point_of_application"], str)
                components += 9 - 3 * is_point_of_application_str

        return components


    def check_segment_names(self, segment_names: StrTuple) -> None:
        attributes = ["in_global", "torque_in_global", "translational_in_global", "in_local", "torque_in_local"]
        wrong_segments = []
        for attr in attributes:
            for force_name , force in getattr(self, attr).items():
                if force["segment"] not in segment_names:
                    wrong_segments.append(force["segment"])

        if wrong_segments:
            raise ValueError(
                f"Segments {wrong_segments} specified in the external forces are not in the model."
                f" Available segments are {segment_names}."
            )

    def check_all_string_points_of_application(self, model_points_of_application: Str) -> None:
        attributes = ["in_global", "translational_in_global", "in_local"]
        wrong_points_of_application = []
        for attr in attributes:
            for segment, force in getattr(self, attr).items():
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

    # Specific functions for adding each force type to improve readability
    @staticmethod
    def add_global_force(biorbd_external_forces, segment: str, force: CXOrDMOrNpArray, point_of_application: CXOrDMOrNpArray):
        biorbd_external_forces.add(segment, force, point_of_application)

    @staticmethod
    def add_torque_global(biorbd_external_forces, segment: str, torque: CXOrDMOrNpArray, _):
        biorbd_external_forces.add(segment, vertcat(torque, MX([0, 0, 0])), MX([0, 0, 0]))

    @staticmethod
    def add_translational_global(biorbd_external_forces, segment: str, force: CXOrDMOrNpArray, point_of_application: CXOrDMOrNpArray):
        biorbd_external_forces.addTranslationalForce(force, segment, point_of_application)

    @staticmethod
    def add_local_force(biorbd_external_forces, segment: str, force, point_of_application):
        biorbd_external_forces.addInSegmentReferenceFrame(segment, force, point_of_application)

    @staticmethod
    def add_torque_local(biorbd_external_forces, segment: str, torque, _):
        biorbd_external_forces.addInSegmentReferenceFrame(segment, vertcat(torque, MX([0, 0, 0])), MX([0, 0, 0]))


class ExternalForceSetTimeSeries(ExternalForceSetCommon):
    """
    A class to manage external forces applied to a set of segments over a series of frames.

    Attributes
    ----------
    _nb_frames : Int
        The number of frames in the time series.
    in_global : AnyDict
        Dictionary to store global external forces for each segment.
    torque_in_global : AnyDict
        Dictionary to store global torques for each segment.
    translational_in_global : AnyDict
        Dictionary to store global translational forces for each segment.
    in_local : AnyDict
        Dictionary to store local external forces for each segment.
    torque_in_local : AnyDict
        Dictionary to store local torques for each segment.
    _bind : Bool
        Flag to indicate if the external forces are binded and cannot be modified.
    """

    def __init__(self, nb_frames: Int):
        """
        Initialize the ExternalForceSetTimeSeries with the number of frames.

        Parameters
        ----------
        nb_frames : int
            The number of frames in the time series.
        """
        self._nb_frames = nb_frames
        super().__init__()

    @property
    def nb_frames(self) -> Int:
        return self._nb_frames

    def _check_values_frame_shape(self, values: NpArray) -> None:
        if values.shape[1] != self._nb_frames:
            raise ValueError(
                f"External forces must have the same number of columns as the number of shooting points, "
                f"got {values.shape[1]} instead of {self._nb_frames}"
            )

    def add(self, force_name: str, segment: str, values: NpArray, point_of_application: NpArrayOptional | Str = None):
        self._check_if_can_be_modified()
        if values.shape[0] != 6:
            raise ValueError(f"External forces must have 6 rows, got {values.shape[0]}")
        self._check_values_frame_shape(values)

        point_of_application = np.zeros((3, self._nb_frames)) if point_of_application is None else point_of_application
        self._check_point_of_application(point_of_application)

        self.in_global[force_name] = {"segment": segment, "values": values, "point_of_application": point_of_application}

    def add_torque(self, force_name: str, segment: str, values: NpArray):
        self._check_if_can_be_modified()
        if values.shape[0] != 3:
            raise ValueError(f"External torques must have 3 rows, got {values.shape[0]}")
        self._check_values_frame_shape(values)

        self.torque_in_global[force_name] = {"segment": segment, "values": values, "point_of_application": None}

    def add_translational_force(
        self, force_name: str, segment: str, values: NpArray, point_of_application_in_local: NpArrayOptional | Str = None
    ):
        self._check_if_can_be_modified()
        if values.shape[0] != 3:
            raise ValueError(f"External forces must have 3 rows, got {values.shape[0]}")
        self._check_values_frame_shape(values)

        point_of_application_in_local = (
            np.zeros((3, self._nb_frames)) if point_of_application_in_local is None else point_of_application_in_local
        )
        self._check_point_of_application(point_of_application_in_local)

        self.translational_in_global[force_name] = {"segment": segment, "values": values, "point_of_application": point_of_application_in_local}

    def add_in_segment_frame(
        self, force_name: str, segment: str, values: NpArray, point_of_application_in_local: NpArrayOptional | Str = None
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
        self.in_local[force_name] = {"segment": segment, "values": values, "point_of_application": point_of_application_in_local}

    def add_torque_in_segment_frame(self, force_name:str, segment: str, values: np.ndarray):
        self._check_if_can_be_modified()
        if values.shape[0] != 3:
            raise ValueError(f"External torques must have 3 rows, got {values.shape[0]}")
        self._check_values_frame_shape(values)

        self.torque_in_local[force_name] = {"segment": segment, "values": values, "point_of_application": None}

    def _check_point_of_application(self, point_of_application: NpArray | Str) -> None:
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

    def to_numerical_time_series(self) -> NpArray:
        """Convert the external forces to a numerical time series"""
        fext_numerical_time_series = np.zeros((self.nb_external_forces_components, 1, self.nb_frames + 1))

        symbolic_counter = 0
        for attr in BIOPTIM_TO_VECTOR_MAP.keys():
            for segment, force in getattr(self, attr).items():
                array_point_of_application = isinstance(force["point_of_application"], np.ndarray)

                start = symbolic_counter
                stop = symbolic_counter + BIOPTIM_TO_VECTOR_MAP[attr]
                force_slicer = slice(start, stop)
                fext_numerical_time_series[force_slicer, 0, :-1] = force["values"]

                if array_point_of_application:
                    poa_slicer = slice(stop, stop + 3)
                    fext_numerical_time_series[poa_slicer, 0, :-1] = force["point_of_application"]

                symbolic_counter = stop + 3 if array_point_of_application else stop

        return fext_numerical_time_series


class ExternalForceSetVariables(ExternalForceSetCommon):
    """
    A class to manage optimized external forces applied to a set of segments.

    Attributes
    ----------
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
    """

    def add(self, force_name:str, segment: str, use_point_of_application: bool = False):
        """
        Add moments XYZ, forces XYZ (and the point of application if requested) in the global reference frame.
        """
        self._check_if_can_be_modified()
        point_of_application = MX.sym(f"point_of_application_{force_name}_{segment}", 3, 1) if use_point_of_application else np.zeros((3, 1))
        moments_forces = MX.sym(f"moments_forces_{force_name}_{segment}", 6, 1)
        self.in_global[force_name] = {"segment": segment, "force": moments_forces, "point_of_application": point_of_application}

    def add_torque(self, force_name:str, segment: str):
        """
        Add moments XYZ in the global reference frame.
        """
        self._check_if_can_be_modified()
        moments = MX.sym(f"moments_{force_name}_{segment}", 3, 1)
        self.torque_in_global[force_name] = {"segment": segment, "force": moments, "point_of_application": None}

    def add_translational_force(
        self, force_name:str, segment: str, use_point_of_application_in_local: bool = False
    ):
        """
        Add forces XYZ (and the point of application if requested) in the local reference frame.
        """
        self._check_if_can_be_modified()
        point_of_application_in_local = MX.sym(f"point_of_application_in_local_{force_name}_{segment}", 3, 1) if use_point_of_application_in_local else np.zeros((3, 1))
        forces = MX.sym(f"forces_{force_name}_{segment}", 3, 1)
        self.translational_in_global[force_name] = {"segment": segment, "force": forces, "point_of_application": point_of_application_in_local}

    def add_in_segment_frame(
        self, force_name:str, segment: str, use_point_of_application_in_local: bool = False
    ):
        """
        Add moments XYZ, forces XYZ (and the point of application if requested) in the local reference frame.
        """
        self._check_if_can_be_modified()
        point_of_application_in_local = MX.sym(f"point_of_application_in_local_{force_name}_{segment}", 3, 1) if use_point_of_application_in_local else np.zeros((3, 1))
        moments_forces = MX.sym(f"moments_forces_{force_name}_{segment}", 6, 1)
        self.in_local[force_name] = {"segment": segment, "force": moments_forces, "point_of_application": point_of_application_in_local}

    def add_torque_in_segment_frame(self, force_name:str, segment: str):
        """
        Add moments XYZ in the local reference frame.
        """
        self._check_if_can_be_modified()
        moments = MX.sym(f"moments_{force_name}_{segment}", 3, 1)
        self.torque_in_local[force_name] = {"segment": segment, "force": moments, "point_of_application": None}

    def to_mx(self):
        """Convert the external forces to an MX vector"""
        fext_numerical_time_series = MX.zeros((self.nb_external_forces_components, 1))

        symbolic_counter = 0
        for attr in BIOPTIM_TO_VECTOR_MAP.keys():
            for segment, forces in getattr(self, attr).items():
                for force in forces:
                    array_point_of_application = not isinstance(force["point_of_application"], np.ndarray)

                    start = symbolic_counter
                    stop = symbolic_counter + BIOPTIM_TO_VECTOR_MAP[attr]
                    force_slicer = slice(start, stop)
                    fext_numerical_time_series[force_slicer, 0, :-1] = force["values"]

                    if array_point_of_application:
                        poa_slicer = slice(stop, stop + 3)
                        fext_numerical_time_series[poa_slicer, 0, :-1] = force["point_of_application"]

                    symbolic_counter = stop + 3 if array_point_of_application else stop

        return fext_numerical_time_series
