from ..limits.path_conditions import Bounds
from ..misc.mapping import BiMapping, BiMappingList


def _dof_mapping(key, model, mapping: BiMapping = None) -> dict:
    if key == "q":
        if model.nb_quaternions > 0 and mapping is not None:
            if "q" in mapping and "qdot" not in mapping:
                raise RuntimeError(
                    "It is not possible to provide a q_mapping but not a qdot_mapping if the model have quaternion"
                )
        return _var_mapping(key, range(model.nb_q), mapping)
    elif key == "q_joints":
        if model.nb_quaternions > 0 and mapping is not None:
            if "q_joints" in mapping and "qdot_joints" not in mapping:
                raise RuntimeError(
                    "It is not possible to provide a q_joints_mapping but not a qdot_joints_mapping if the model have quaternion"
                )
        return _var_mapping(key, range(model.nb_q - model.nb_root), mapping)
    elif key == "q_roots":
        return _var_mapping(key, range(model.nb_root), mapping)
    elif key == "qdot":
        if model.nb_quaternions > 0 and mapping is not None:
            if "qdot" in mapping and "q" not in mapping:
                raise RuntimeError(
                    "It is not possible to provide a qdot_mapping but not a q_mapping if the model have quaternion"
                )
        return _var_mapping(key, range(model.nb_qdot), mapping)
    elif key == "qdot_joints":
        if model.nb_quaternions > 0 and mapping is not None:
            if "qdot_joints" in mapping and "q_joints" not in mapping:
                raise RuntimeError(
                    "It is not possible to provide a qdot_joints_mapping but not a q_joints_mapping if the model have quaternion"
                )
        return _var_mapping(key, range(model.nb_qdot - model.nb_root), mapping)
    elif key == "qdot_roots":
        return _var_mapping(key, range(model.nb_root), mapping)
    elif key == "qddot":
        return _var_mapping(key, range(model.nb_qdot), mapping)
    elif key == "qddot_joints":
        return _var_mapping(key, range(model.nb_qdot - model.nb_root), mapping)
    else:
        raise NotImplementedError("Wrong dof mapping")


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
