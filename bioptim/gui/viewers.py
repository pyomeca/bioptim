from typing import Any

import biorbd_casadi as biorbd
import numpy as np
from scipy.interpolate import interp1d

from ..limits.objective_functions import ObjectiveFcn
from ..misc.enums import Node
from ..misc.utils import check_version
from ..models.biorbd.biorbd_model import BiorbdModel


def animate_with_bioviz_for_loop(
    ocp: "OptimalControlProgram",
    data_to_animate: "Solution",
    show_now: bool = True,
    tracked_markers: dict[int, list[str]] | list[str] = None,
    **kwargs,
):
    """
    Animate the solution(s) using bioviz

    Parameters
    ----------
    ocp: OptimalControlProgram
        The optimal control program
    data_to_animate: Solution| List[Solution]
        The solution(s) to animate
    show_now: bool
        If the animation should be shown immediately
    tracked_markers: dict[int, list[str]] | list[str]
        The markers to track
    kwargs
        Any other parameters to pass to bioviz
    """

    all_bioviz = []
    for i, data in enumerate(data_to_animate):
        all_bioviz.append(
            ocp.nlp[i].model.animate(
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
    show_now: bool = True,
    tracked_markers: list[np.ndarray] = None,
    **kwargs: Any,
) -> None | list:
    try:
        import bioviz
    except ModuleNotFoundError:
        raise RuntimeError("bioviz must be install to animate the model")

    check_version(bioviz, "2.0.0", "2.4.0")

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


def play_bioviz_animation(all_bioviz: list) -> None:
    """Play the animation of the list of bioviz objects"""
    b_is_visible = [True] * len(all_bioviz)
    while sum(b_is_visible):
        for i, b in enumerate(all_bioviz):
            if b.vtk_window.is_active:
                b.update()
            else:
                b_is_visible[i] = False
    return None


def animate_with_pyorerun(
    solution: "SolutionData",
    show_now: bool = True,
    tracked_markers: list[np.ndarray] = None,
    models: BiorbdModel | list[BiorbdModel] = None,
    **kwargs: Any,
) -> None:
    try:
        import pyorerun
    except ModuleNotFoundError:
        raise RuntimeError("pyorerun must be install to animate the model")

    if not isinstance(solution, (list, tuple)):
        solution = [solution]

    if tracked_markers is None:
        tracked_markers = [None] * len(solution)
    prerun = pyorerun.MultiPhaseRerun()

    for idx_phase, (data, model, tm) in enumerate(zip(solution, models, tracked_markers)):

        if "q_roots" in data and "q_joints" in data:
            try:
                data["q"] = np.vstack((data["q_roots"], data["q_joints"]))
            except:
                raise NotImplementedError(
                    "Found q_roots and q_joints in the solution. This is not supported yet with animation in pyorerun"
                )

        prerun.add_phase(t_span=data["time"], phase=idx_phase)

        if not isinstance(model, biorbd.Model):
            raise NotImplementedError(
                f"Animation is only implemented for biorbd models. Got {model.__class__.__name__}"
            )

        biorbd_model = pyorerun.BiorbdModel.from_biorbd_object(model)

        prerun.add_animated_model(
            biorbd_model,
            data["q"],
            tracked_markers=tm if tm is not None else None,
            phase=idx_phase,
        )

    prerun.rerun(notebook=not show_now)


def _prepare_tracked_markers_for_animation(
    nlps: list["NonLinearProgram", ...], n_shooting: int = None
) -> list[np.ndarray, ...]:
    """Prepare the markers which are tracked to the animation"""

    all_tracked_markers = []

    for phase, nlp in enumerate(nlps):
        n_frames = sum(nlp.ns) + 1 if n_shooting is None else n_shooting + 1

        n_states_nodes = nlp.n_states_nodes

        tracked_markers = None
        for objective in nlp.J:
            if objective.target is not None:
                if objective.type in (
                    ObjectiveFcn.Mayer.TRACK_MARKERS,
                    ObjectiveFcn.Lagrange.TRACK_MARKERS,
                ) and objective.node[0] in (Node.ALL, Node.ALL_SHOOTING):
                    tracked_markers = np.full((3, nlp.model.nb_markers, n_states_nodes), np.nan)

                    for i in range(len(objective.rows)):
                        tracked_markers[objective.rows[i], objective.cols, :] = objective.target[i, :, :]

                    missing_row = np.where(np.isnan(tracked_markers))[0]
                    if missing_row.size > 0:
                        tracked_markers[missing_row, :, :] = 0

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
