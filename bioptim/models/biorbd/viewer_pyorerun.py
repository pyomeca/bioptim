import biorbd_casadi as biorbd
import numpy as np
import pyorerun
from typing import Any

from .biorbd_model import BiorbdModel
from .multi_biorbd_model import MultiBiorbdModel
from ...optimization.solution.solution_data import SolutionMerge


def animate_with_pyorerun(
    ocp,
    solution,
    show_now: bool,
    show_tracked_markers: bool,
    **kwargs: Any,
) -> None:
    try:
        import pyorerun
    except ModuleNotFoundError:
        raise RuntimeError("pyorerun must be install to animate the model")

    data_to_animate, models, tracked_markers = prepare_pyorerun_animation(ocp, solution, show_now, show_tracked_markers)
    launch_rerun(data_to_animate, show_now, tracked_markers, models, **kwargs)


def prepare_pyorerun_animation(ocp, solution, show_now=True, show_tracked_markers=True) -> tuple[list, list, None]:
    """Extract data from the solution to isolate the data for each phase and each model"""
    n_phases = ocp.n_phases
    data_to_animate = solution.decision_states(to_merge=SolutionMerge.NODES)
    if n_phases == 1:
        data_to_animate = [data_to_animate]

    data_to_animate = set_time(solution, n_phases, data_to_animate)

    models = []
    for i, nlp in enumerate(ocp.nlp):
        if isinstance(nlp.model, MultiBiorbdModel):
            data_to_animate, models = set_data_for_multibiorbd_model(nlp, data_to_animate, models, i)
        else:
            models += [nlp.model.model]

    if show_tracked_markers:
        raise NotImplementedError(
            "Tracking markers is not implemented for pyorerun. "
            "Set show_tracked_markers to False such that sol.animate(show_tracked_markers=False)."
        )
        # TODO : Implement tracking markers for pyorerun
        #  with the _prepare_tracked_markers_for_animation in viewer_utils.py
    else:
        tracked_markers = None

    return data_to_animate, models, tracked_markers


def set_time(solution, n_phases, data_to_animate):
    """Set the time in the data_to_animate, handles multiple phases"""
    time = (
        [np.concatenate(solution.decision_time()).squeeze()]
        if n_phases == 1
        else [np.concatenate(t).squeeze() for t in solution.decision_time()]
    )

    for i in range(n_phases):
        data_to_animate[i]["time"] = time[i]

    return data_to_animate


def set_data_for_multibiorbd_model(nlp, data_to_animate: list, models: list, i: int):
    """Duplicate the data for each model in the MultiBiorbdModel and models"""
    models += [model for model in nlp.model.models]
    temp_data_animate = [data_to_animate[i].copy() for _ in range(nlp.model.nb_models)]
    for j, model in enumerate(nlp.model.models):
        for key in data_to_animate[i].keys():
            if key == "time":
                continue
            index = nlp.model.variable_index(key, j)
            temp_data_animate[j][key] = temp_data_animate[j][key][index, :]

    data_to_animate[i] = temp_data_animate[0]
    data_to_animate += temp_data_animate[1:]

    return data_to_animate, models


def launch_rerun(
    solution: "SolutionData",
    show_now: bool = True,
    tracked_markers: list[np.ndarray] = None,
    models: BiorbdModel | list[BiorbdModel] = None,
    **kwargs: Any,
):
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
