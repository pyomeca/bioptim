import numpy as np
from scipy.interpolate import interp1d

from ...limits.objective_functions import ObjectiveFcn
from ...misc.enums import Node


def _prepare_tracked_markers_for_animation(
    nlps: list["NonLinearProgram", ...], n_shooting: int = None
) -> list[np.ndarray, ...]:
    """
    Prepare the markers which are tracked to the animation

    Returns
    -------
    list[np.ndarray, ...]
        The markers to be tracked for each phase
    """

    all_tracked_markers = []

    for phase, nlp in enumerate(nlps):
        n_frames = nlp.ns + 1 if n_shooting is None else n_shooting + 1

        n_states_nodes = nlp.n_states_nodes

        tracked_markers = None
        for objective in nlp.J:

            objective_has_a_target = objective.target is not None
            objective_is_tracking_markers = objective.type in (
                ObjectiveFcn.Mayer.TRACK_MARKERS,
                ObjectiveFcn.Lagrange.TRACK_MARKERS,
            )
            objective_is_tracking_all_nodes = objective.node[0] in (Node.ALL, Node.ALL_SHOOTING)

            if objective_has_a_target and objective_is_tracking_markers and objective_is_tracking_all_nodes:
                tracked_markers = np.zeros((3, nlp.model.nb_markers, n_states_nodes))

                for i, row in enumerate(objective.rows):
                    tracked_markers[row, objective.cols, :] = objective.target[i, ...]

        # interpolation
        if n_frames > 0 and tracked_markers is not None:
            x = np.linspace(0, n_states_nodes - 1, n_states_nodes)
            xnew = np.linspace(0, n_states_nodes - 1, n_frames)
            f = interp1d(x, tracked_markers, kind="cubic")
            tracked_markers = f(xnew)

        all_tracked_markers.append(tracked_markers)

    return all_tracked_markers


def _check_models_comes_from_same_super_class(all_nlp: list["NonLinearProgram", ...]):
    """Check that all the models comes from the same super class"""
    for i, nlp in enumerate(all_nlp):
        model_super_classes = nlp.model.__class__.mro()[:-1]  # remove object class
        nlps = all_nlp.copy()
        del nlps[i]
        for j, sub_nlp in enumerate(nlps):
            if not any([isinstance(sub_nlp.model, super_class) for super_class in model_super_classes]):
                raise RuntimeError(
                    f"The animation is only available for compatible models. "
                    f"Here, the model of phase {i} is of type {nlp.model.__class__.__name__} and the model of "
                    f"phase {j + 1 if i < j else j} is of type {sub_nlp.model.__class__.__name__} and "
                    f"they don't share the same super class."
                )
