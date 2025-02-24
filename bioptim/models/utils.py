from ..limits.path_conditions import Bounds
from ..misc.mapping import BiMapping, BiMappingList


def _dof_mapping(key, model, mapping: BiMapping = None) -> dict:
    has_quaternion_and_mapping = model.nb_quaternions > 0 and mapping is not None
    if has_quaternion_and_mapping:
        _check_quaternion_mapping(key, mapping, model)

    ranges_map = {
        "q": range(model.nb_q),
        "q_joints": range(model.nb_q - model.nb_root),
        "q_roots": range(model.nb_root),
        "qdot": range(model.nb_qdot),
        "qdot_joints": range(model.nb_qdot - model.nb_root),
        "qdot_roots": range(model.nb_root),
        "qddot": range(model.nb_qdot),
        "qddot_joints": range(model.nb_qdot - model.nb_root),
    }

    if key in ranges_map:
        return _var_mapping(key, ranges_map[key], mapping)
    else:
        raise NotImplementedError("Wrong dof mapping")


def _check_quaternion_mapping(key, mapping, model):
    required_mappings = {"q": "qdot", "q_joints": "qdot_joints", "qdot": "q", "qdot_joints": "q_joints"}
    if key in mapping and required_mappings[key] not in mapping:
        raise RuntimeError(
            f"It is not possible to provide a {key}_mapping but not a {required_mappings[key]}_mapping if the model has quaternion"
        )


def _var_mapping(key, range_for_mapping, mapping: BiMapping = None) -> dict:
    """
    This function returns a standard mapping for the variable key if None.
    """
    if mapping is None:
        mapping = {}
    if key not in mapping:
        mapping[key] = BiMapping(range_for_mapping, range_for_mapping)
    return mapping


def bounds_from_ranges(model, key: str, mapping: BiMapping | BiMappingList = None) -> Bounds:
    """
    Generate bounds from the ranges of the model

    Parameters
    ----------
    model: bio_model
        such as BiorbdModel or MultiBiorbdModel
    key: str | list[str]
        The variables to generate the bounds from, such as "q", "qdot", "qddot", or ["q", "qdot"],
    mapping: BiMapping | BiMappingList
        The mapping to use to generate the bounds. If None, the default mapping is built

    Returns
    -------
    Bounds
        The bounds generated from the ranges of the model
    """

    mapping_tp = _dof_mapping(key, model, mapping)[key]
    ranges = model.ranges_from_model(key)

    x_min = [ranges[i].min() for i in mapping_tp.to_first.map_idx]
    x_max = [ranges[i].max() for i in mapping_tp.to_first.map_idx]
    return Bounds(key, min_bound=x_min, max_bound=x_max)
