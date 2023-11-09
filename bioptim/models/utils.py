from ..limits.path_conditions import Bounds
from ..misc.mapping import BiMapping, BiMappingList


def _dof_mapping(key, model, mapping: BiMapping = None) -> dict:
    if key == "q":
        return _q_mapping(model, mapping)
    elif key == "qdot":
        return _qdot_mapping(model, mapping)
    elif key == "qddot":
        return _qddot_mapping(model, mapping)
    else:
        raise NotImplementedError("Wrong dof mapping")


def _q_mapping(model, mapping: BiMapping = None) -> dict:
    """
    This function returns a standard mapping for the q states if None
    and checks if the model has quaternions
    """
    if mapping is None:
        mapping = {}
    if model.nb_quaternions > 0:
        if "q" in mapping and "qdot" not in mapping:
            raise RuntimeError(
                "It is not possible to provide a q_mapping but not a qdot_mapping if the model have quaternion"
            )
        elif "q" not in mapping and "qdot" in mapping:
            raise RuntimeError(
                "It is not possible to provide a qdot_mapping but not a q_mapping if the model have quaternion"
            )
    if "q" not in mapping:
        mapping["q"] = BiMapping(range(model.nb_q), range(model.nb_q))
    return mapping


def _qdot_mapping(model, mapping: BiMapping = None) -> dict:
    """
    This function returns a standard mapping for the qdot states if None
    and checks if the model has quaternions
    """
    if mapping is None:
        mapping = {}
    if "qdot" not in mapping:
        mapping["qdot"] = BiMapping(range(model.nb_qdot), range(model.nb_qdot))

    return mapping


def _qddot_mapping(model, mapping: BiMapping = None) -> dict:
    """
    This function returns a standard mapping for the qddot states if None
    and checks if the model has quaternions
    """
    if mapping is None:
        mapping = {}
    if "qddot" not in mapping:
        mapping["qddot"] = BiMapping(range(model.nb_qddot), range(model.nb_qddot))

    return mapping


def bounds_from_ranges(model, key: str, mapping: BiMapping | BiMappingList = None) -> Bounds:
    """
    Generate bounds from the ranges of the model

    Parameters
    ----------
    model: bio_model
        such as BiorbdModel or MultiBiorbdModel
    key: str | list[str, ...]
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
