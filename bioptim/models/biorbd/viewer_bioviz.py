import numpy as np
from typing import Any

from .biorbd_model import BiorbdModel
from .viewer_utils import _prepare_tracked_markers_for_animation
from ...limits.objective_functions import ObjectiveFcn
from ...misc.enums import Node
from ...misc.utils import check_version
from ...misc.parameters_types import Int, Bool, NpArray, AnyList, AnyListOptional


def animate_with_bioviz_for_loop(
    ocp: "OptimalControlProgram",
    solution: "Solution",
    show_now: Bool = True,
    show_tracked_markers: Bool = True,
    n_frames: Int = None,
    **kwargs,
):
    """
    Animate the solution(s) using bioviz

    Parameters
    ----------
    ocp: OptimalControlProgram
        The optimal control program
    solution: Solution| List[Solution]
        The solution(s) to animate
    show_now: bool
        If the animation should be shown immediately
    show_tracked_markers: bool
        If the tracked markers should be shown
    n_frames: int
        The number of frames to interpolate
    kwargs
        Any other parameters to pass to bioviz
    """
    n_frames = count_in_track_markers(ocp.nlp, n_frames)
    data_to_animate = interpolate_data(solution, ocp, n_frames)
    tracked_markers = get_tracked_markers(show_tracked_markers, ocp, n_frames)

    all_bioviz = []
    for i, data in enumerate(data_to_animate):
        all_bioviz.append(
            animate_with_bioviz(
                ocp,
                solution=data,
                show_now=show_now,
                tracked_markers=tracked_markers,
                **kwargs,
            )
        )

    return all_bioviz


def animate_with_bioviz(
    ocp,
    solution: "SolutionData",
    show_now: Bool = True,
    tracked_markers: list[NpArray] = None,
    **kwargs: Any,
) -> AnyListOptional:
    try:
        import bioviz
    except ModuleNotFoundError:
        raise RuntimeError("bioviz must be install to animate the model")

    check_version(bioviz, "2.4.0", "2.5.0")

    if "q_roots" in solution and "q_joints" in solution:
        states = np.vstack((solution["q_roots"], solution["q_joints"]))
    else:
        states = solution["q"]

    if not isinstance(states, (list, tuple)):
        states = [states]

    if tracked_markers is None:
        tracked_markers = [None] * len(states)

    all_bioviz = []
    for idx_phase, data in enumerate(states):
        if not isinstance(ocp.nlp[idx_phase].model, BiorbdModel):
            raise NotImplementedError("Animation is only implemented for biorbd models")

        # This calls each of the function that modify the internal dynamic model based on the parameters
        nlp = ocp.nlp[idx_phase]

        # noinspection PyTypeChecker
        biorbd_model: BiorbdModel = nlp.model

        all_bioviz.append(bioviz.Viz(biorbd_model.path, **kwargs))
        all_bioviz[-1].load_movement(ocp.nlp[idx_phase].variable_mappings["q"].to_second.map(solution["q"]))

        if "q_roots" in solution and "q_joints" in solution:
            # TODO: Fix the mapping for this case
            raise NotImplementedError("Mapping is not implemented for this case")
            q = data
        else:
            q = ocp.nlp[idx_phase].variable_mappings["q"].to_second.map(data)
        all_bioviz[-1].load_movement(q)

        if tracked_markers[idx_phase] is not None:
            all_bioviz[-1].load_experimental_markers(tracked_markers[idx_phase])

    return play_bioviz_animation(all_bioviz) if show_now else all_bioviz


def play_bioviz_animation(all_bioviz: AnyList) -> None:
    """Play the animation of the list of bioviz objects"""
    b_is_visible = [True] * len(all_bioviz)
    while sum(b_is_visible):
        for i, b in enumerate(all_bioviz):
            if b.vtk_window.is_active:
                b.update()
            else:
                b_is_visible[i] = False
    return None


def count_in_track_markers(nlps, n_frames) -> Int:
    """Ipuch, Legacy Code: I'm very not sure what it does if anyone knows, fix the name of the function"""
    for idx_phase in range(len(nlps)):
        for objective in nlps[idx_phase].J:
            if objective.target is not None:
                if objective.type in (
                    ObjectiveFcn.Mayer.TRACK_MARKERS,
                    ObjectiveFcn.Lagrange.TRACK_MARKERS,
                ) and objective.node[0] in (Node.ALL, Node.ALL_SHOOTING):
                    n_frames += objective.target.shape[2]
                    break
    return n_frames


def interpolate_data(solution, ocp, n_frames) -> AnyList:
    """Interpolate the data to be animated"""
    data_to_animate = []
    if n_frames == 0:
        try:
            data_to_animate = [solution.interpolate(sum([nlp.ns for nlp in ocp.nlp]) + 1)]
        except ValueError:
            data_to_animate = solution.interpolate([nlp.ns for nlp in ocp.nlp])
    elif n_frames > 0:
        data_to_animate = solution.interpolate(n_frames)
        if not isinstance(data_to_animate, list):
            data_to_animate = [data_to_animate]

    return data_to_animate


def get_tracked_markers(show_tracked_markers, ocp, n_frames):
    """Get the tracked markers if needed"""
    if show_tracked_markers and len(ocp.nlp) == 1:
        return _prepare_tracked_markers_for_animation(ocp.nlp, n_shooting=n_frames)
    elif show_tracked_markers and len(ocp.nlp) > 1:
        raise NotImplementedError(
            "Tracking markers is not implemented for multiple phases. "
            "Set show_tracked_markers to False such that sol.animate(show_tracked_markers=False)."
        )
    else:
        return [None for _ in range(len(ocp.nlp))]
